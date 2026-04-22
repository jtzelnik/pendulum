#pragma once               // guard against multiple inclusion
#include "../common/protocol.h"  // StatePacket (wire struct shared with RL client)
#include "pendulum_hw.h"         // EncoderState, pin constants, kinematics constants
#include <chrono>                // std::chrono::steady_clock for wall-clock timestamps

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
