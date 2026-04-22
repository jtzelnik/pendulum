#pragma once               // guard against multiple inclusion
#include "pendulum_hw.h"   // EncoderState, pin constants, kinematics constants
#include <chrono>          // std::chrono::steady_clock for wall-clock timestamps
#include <cstdint>         // int64_t

// Wire-ready observation packet transmitted to the client PC each control tick.
// Field order and types are fixed — ZeroMQ sends this struct as raw bytes.
// Contains the four DRL states plus an episode status flag.
// Hardware limit logic is enforced inside the LLI; the status field tells the
// client when an episode has been terminated and re-homing has begun.
struct StatePacket {
    int64_t  timestamp_us;    // microseconds since steady_clock epoch — client uses consecutive values to compute actual dt
    double   x;               // carriage position      (m,     zero = rail centre after homing)
    double   x_dot;           // carriage velocity      (m/s,   exponential moving average filtered)
    double   theta;           // pendulum angle         (rad,   zero = hanging down after homing)
    double   theta_dot;       // pendulum angular vel.  (rad/s, exponential moving average filtered)
    uint8_t  episode_status;  // 0 = running, 1 = limit sensor hit, 2 = |theta_dot| exceeded 14 rad/s
};

// Maintains inter-call filter state for the velocity EMA.
// One instance lives for the lifetime of the control loop; call update() once per tick.
struct StateEstimator {
    StateEstimator() = default;   // zero-initialises all filter state

    // Reads both encoders, computes EMA-filtered velocities using actual elapsed dt,
    // and returns a fully populated StatePacket ready for transmission.
    StatePacket update(const EncoderState& enc_carriage,
                       const EncoderState& enc_pendulum);

private:
    long long prev_c1{0}, prev_c2{0};          // encoder counts from the previous tick
    double    x_dot{0.0}, th_dot{0.0};         // current filtered velocity estimates
    std::chrono::steady_clock::time_point last_time{};   // wall-clock time of the previous update call
    bool      first_call{true};                // true until the first update() completes; uses nominal DT on first tick
};
