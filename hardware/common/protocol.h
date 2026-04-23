#pragma once    // prevent this header from being included more than once per translation unit
#include <cstdint>  // int64_t, int32_t, uint8_t — fixed-width types so sizes are the same on every platform

// ── Wire protocol overview ────────────────────────────────────────────────────
// The LLI (Raspberry Pi) and the RL client (PC) exchange raw C structs over
// ZeroMQ TCP sockets with no serialisation layer. Both sides must use identical
// struct layouts; this header is the single source of truth.
//
// Both machines are little-endian (ARM Cortex-A72 and x86-64), so byte order
// is not a concern, but the Python side uses '<' (explicit little-endian) in
// struct.pack/unpack to make the assumption explicit.
//
// StatePacket  — LLI → PC, published once per 50 Hz tick.   48 bytes.
// MotorCommand — PC  → LLI, pushed by the PC each tick.      8 bytes.

// ── StatePacket ───────────────────────────────────────────────────────────────
// One packet is published by the LLI every 20 ms. The PC receives it and
// extracts the state vector for the RL model.
//
// Memory layout (C struct, little-endian):
//   Offset  0: int64  timestamp_us   (8 bytes)
//   Offset  8: double x              (8 bytes)
//   Offset 16: double x_dot          (8 bytes)
//   Offset 24: double theta          (8 bytes)
//   Offset 32: double theta_dot      (8 bytes)
//   Offset 40: uint8  episode_status (1 byte)
//   Offset 41: 7 padding bytes       (compiler aligns sizeof to 8)
//   Total: 48 bytes
//
// Python mirror: '<qddddB7x' (see hardware/rl/protocol.py)
struct StatePacket {
    int64_t  timestamp_us;    // wall-clock time of this tick in microseconds since steady_clock epoch
    double   x;               // carriage position in metres; 0 = rail centre after homing
    double   x_dot;           // carriage velocity in m/s; Butterworth-filtered to reduce encoder noise
    double   theta;           // pendulum angle in radians; 0 = hanging straight down after homing
    double   theta_dot;       // pendulum angular velocity in rad/s; Butterworth-filtered
    uint8_t  episode_status;  // episode state code — see values below:
                              //   0 = running normally
                              //   1 = proximity limit sensor triggered (carriage hit a stop)
                              //   2 = |theta_dot| exceeded 14 rad/s safety threshold
                              //   3 = homing started in response to the PC's request_home flag
};

// ── MotorCommand ──────────────────────────────────────────────────────────────
// Sent by the PC to the LLI each tick. The LLI drains the queue and applies
// only the most recent command, so stale commands never accumulate.
//
// Memory layout:
//   Offset 0: int32  duty         (4 bytes)
//   Offset 4: uint8  estop        (1 byte)
//   Offset 5: uint8  request_home (1 byte)
//   Offset 6: 2 padding bytes     (aligns sizeof to 4)
//   Total: 8 bytes
//
// Python mirror: '<iBB2x' (see hardware/rl/protocol.py)
struct MotorCommand {
    int32_t  duty;          // signed PWM duty: +255 = full right, -255 = full left, 0 = coast
    uint8_t  estop;         // emergency stop: any non-zero value → LLI immediately stops the motor
    uint8_t  request_home;  // re-home request: non-zero → LLI runs a homing cycle before the next episode
};
