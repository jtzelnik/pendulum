#pragma once    // prevent this header from being included more than once per translation unit
#include <cstdint>  // int64_t, int32_t, uint8_t — fixed-width types that match Python struct sizes exactly

// ── StatePacket ───────────────────────────────────────────────────────────────
// Wire-ready observation packet published by the LLI over ZeroMQ each control tick.
// Field order, types, and padding are fixed — ZeroMQ sends the raw struct bytes
// with no serialisation layer, so the layout must match exactly on both ends.
// C sizeof = 48 bytes: 8(i64) + 4×8(f64) + 1(u8) + 7(pad to align to 8).
// Python mirror in hardware/rl/protocol.py uses format string '<qddddB7x'.
struct StatePacket {
    int64_t  timestamp_us;    // wall-clock time of this tick in microseconds since steady_clock epoch; client uses consecutive values to compute actual dt
    double   x;               // carriage position in metres; zero = rail centre after homing
    double   x_dot;           // carriage velocity in m/s; exponential moving-average filtered
    double   theta;           // pendulum angle in radians; zero = hanging straight down after homing
    double   theta_dot;       // pendulum angular velocity in rad/s; exponential moving-average filtered
    uint8_t  episode_status;  // 0 = episode running normally; 1 = proximity limit sensor triggered; 2 = |theta_dot| exceeded 14 rad/s safety threshold; 3 = homing initiated by client request_home (published once before homing begins)
};

// ── MotorCommand ──────────────────────────────────────────────────────────────
// Command message pushed from the RL client to the LLI each control tick.
// C sizeof = 8 bytes: 4(i32) + 1(u8) + 1(u8) + 2(pad to align to 4).
// Python mirror in hardware/rl/protocol.py uses format string '<iBB2x'.
struct MotorCommand {
    int32_t  duty;          // signed PWM duty cycle: +255 = full speed right, -255 = full speed left, 0 = coast
    uint8_t  estop;         // emergency stop flag: any non-zero value commands immediate motor halt
    uint8_t  request_home;  // non-zero: LLI should stop motor and run a homing cycle before the next episode
};
