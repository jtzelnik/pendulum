"""
Wire protocol — Python mirror of hardware/common/protocol.h.

Both the Raspberry Pi (ARM, little-endian) and the client PC (x86-64, little-endian)
store multi-byte numbers the same way, so no byte-swapping is needed. We still use
'<' (explicit little-endian) throughout to make the assumption visible and to
ensure correctness if either machine ever changes.

Struct layouts must match the C sizeof exactly — any mismatch silently corrupts data:

  StatePacket  '<qddddB7x'  — 48 bytes
    q  = int64    (8 bytes)  timestamp_us
    d  = float64  (8 bytes)  x
    d  = float64  (8 bytes)  x_dot
    d  = float64  (8 bytes)  theta
    d  = float64  (8 bytes)  theta_dot
    B  = uint8    (1 byte)   episode_status
    7x = 7 pad bytes         matches the compiler's automatic alignment padding

  MotorCommand '<iBB2x'     —  8 bytes
    i  = int32    (4 bytes)  duty
    B  = uint8    (1 byte)   estop
    B  = uint8    (1 byte)   request_home
    2x = 2 pad bytes         matches the compiler's automatic alignment padding
"""

import struct           # standard-library binary packing and unpacking
from typing import NamedTuple   # NamedTuple: like a tuple but with named fields (pkt.theta instead of pkt[3])

# ── Episode status codes ──────────────────────────────────────────────────────
# These must match the uint8_t values defined in common/protocol.h.
EPISODE_RUNNING        = 0   # normal tick — episode is in progress
EPISODE_LIMIT_HIT      = 1   # proximity sensor fired — carriage hit a rail stop; LLI auto-homes
EPISODE_ANGVEL_EXCEED  = 2   # |theta_dot| > 14 rad/s — too fast to recover; LLI auto-homes
EPISODE_HOMING_STARTED = 3   # LLI acknowledged the PC's request_home; homing is now beginning

# ── StatePacket ───────────────────────────────────────────────────────────────
_STATE_FMT = "<qddddB7x"                         # format string passed to struct.unpack
STATE_SIZE  = struct.calcsize(_STATE_FMT)         # compute byte length at import time (must equal 48)
assert STATE_SIZE == 48, f"StatePacket size mismatch: {STATE_SIZE}"   # crash immediately if the layout is wrong

class StatePacket(NamedTuple):
    """One decoded observation packet received from the LLI over ZeroMQ.

    All velocity fields have already been filtered by the LLI's Butterworth
    filter before transmission; the PC does not need to filter them further.
    """
    timestamp_us:   int    # microseconds since the LLI's steady_clock epoch — useful for inter-packet interval checks
    x:              float  # carriage position in metres (0 = rail centre)
    x_dot:          float  # carriage velocity in m/s (Butterworth-filtered)
    theta:          float  # pendulum angle in radians (0 = hanging straight down)
    theta_dot:      float  # pendulum angular velocity in rad/s (Butterworth-filtered)
    episode_status: int    # 0=running  1=limit hit  2=ang-vel exceeded  3=homing started

# ── MotorCommand ──────────────────────────────────────────────────────────────
_CMD_FMT = "<iBB2x"                              # format string passed to struct.pack
CMD_SIZE  = struct.calcsize(_CMD_FMT)             # compute byte length at import time (must equal 8)
assert CMD_SIZE == 8, f"MotorCommand size mismatch: {CMD_SIZE}"

# ── Decode helper ─────────────────────────────────────────────────────────────
def unpack_state(data: bytes) -> StatePacket:
    """Deserialise raw ZeroMQ message bytes into a StatePacket.

    Slices to STATE_SIZE before unpacking so the function tolerates any
    trailing bytes without raising an error (defensive against future protocol
    extensions that add fields to the end of the struct).
    """
    return StatePacket(*struct.unpack(_STATE_FMT, data[:STATE_SIZE]))

# ── Encode helper ─────────────────────────────────────────────────────────────
def pack_cmd(duty: int, estop: bool = False, request_home: bool = False) -> bytes:
    """Serialise a motor command into raw bytes ready for ZeroMQ send.

    Args:
        duty:         Signed PWM duty: positive = right, negative = left, 0 = coast.
        estop:        True → LLI immediately stops the motor regardless of duty.
        request_home: True → LLI runs a homing cycle before resuming control.
    """
    return struct.pack(_CMD_FMT, int(duty), int(estop), int(request_home))
