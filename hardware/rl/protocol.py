"""
Binary wire protocol — Python mirror of hardware/common/protocol.h.

Both the Raspberry Pi 4 (ARM Cortex-A72) and the client PC (x86-64) are
little-endian, so '<' is used throughout to make the byte order explicit
rather than relying on the native-endian '@' specifier.

Struct layouts (must match C sizeof exactly):
  StatePacket  '<qddddB7x'  — 48 bytes
  MotorCommand '<iB3x'      —  8 bytes

Sizes are verified with assert at import time so a layout mismatch raises
immediately rather than silently corrupting data at runtime.
"""

import struct           # standard-library binary packing/unpacking
from typing import NamedTuple   # lightweight immutable record type for the decoded packet

# Episode status codes — must match uint8_t values in common/protocol.h
EPISODE_RUNNING        = 0   # normal tick
EPISODE_LIMIT_HIT      = 1   # proximity sensor triggered; LLI auto-homes
EPISODE_ANGVEL_EXCEED  = 2   # |theta_dot| > 14 rad/s; LLI auto-homes
EPISODE_HOMING_STARTED = 3   # LLI acknowledged request_home; homing beginning now

# ── StatePacket wire format ───────────────────────────────────────────────────
# '<' = little-endian, standard sizes (no platform-dependent alignment).
# q  = int64  (8 bytes) — timestamp_us
# d  = float64 (8 bytes) — x
# d  = float64 (8 bytes) — x_dot
# d  = float64 (8 bytes) — theta
# d  = float64 (8 bytes) — theta_dot
# B  = uint8  (1 byte)  — episode_status
# 7x = 7 pad bytes      — matches the 7-byte trailing pad the C compiler adds
#                          to align sizeof(StatePacket) to the largest member (8)
_STATE_FMT = "<qddddB7x"                          # format string passed to struct.unpack
STATE_SIZE  = struct.calcsize(_STATE_FMT)          # compute byte length at import time (expected: 48)
assert STATE_SIZE == 48, f"StatePacket size mismatch: {STATE_SIZE}"   # die loudly if layout is wrong

# ── MotorCommand wire format ──────────────────────────────────────────────────
# '<' = little-endian.
# i  = int32  (4 bytes) — duty
# B  = uint8  (1 byte)  — estop
# B  = uint8  (1 byte)  — request_home
# 2x = 2 pad bytes      — matches trailing pad that aligns sizeof(MotorCommand) to 4
_CMD_FMT = "<iBB2x"                              # format string passed to struct.pack
CMD_SIZE  = struct.calcsize(_CMD_FMT)             # compute byte length at import time (expected: 8)
assert CMD_SIZE == 8, f"MotorCommand size mismatch: {CMD_SIZE}"    # die loudly if layout is wrong


# ── StatePacket Python record ─────────────────────────────────────────────────
# NamedTuple gives attribute access (pkt.theta) without the overhead of a full
# dataclass, and is immutable so fields cannot be accidentally mutated after decode.
class StatePacket(NamedTuple):
    """Decoded observation received from the LLI over ZeroMQ."""
    timestamp_us:   int    # wall-clock tick time in microseconds since steady_clock epoch
    x:              float  # carriage position in metres (0 = rail centre)
    x_dot:          float  # carriage velocity in m/s (Butterworth-filtered)
    theta:          float  # pendulum angle in radians (0 = hanging down)
    theta_dot:      float  # pendulum angular velocity in rad/s (Butterworth-filtered)
    episode_status: int    # 0=running  1=limit hit  2=angular-velocity exceeded  3=homing started (request_home ack)


# ── Decode helper ─────────────────────────────────────────────────────────────
def unpack_state(data: bytes) -> StatePacket:
    """Deserialise a raw ZeroMQ message into a StatePacket.

    Slices to STATE_SIZE before unpacking so the function tolerates any
    trailing bytes without raising an error.
    """
    return StatePacket(*struct.unpack(_STATE_FMT, data[:STATE_SIZE]))   # unpack exactly 48 bytes, ignore any trailing data


# ── Encode helper ─────────────────────────────────────────────────────────────
def pack_cmd(duty: int, estop: bool = False, request_home: bool = False) -> bytes:
    """Serialise a motor command into raw bytes ready for ZeroMQ send."""
    return struct.pack(_CMD_FMT, int(duty), int(estop), int(request_home))
