#pragma once               // guard against multiple inclusion
#include "../common/protocol.h"  // StatePacket (wire struct shared with RL client)
#include "pendulum_hw.h"         // EncoderState, pin constants, kinematics constants
#include <chrono>                // std::chrono::steady_clock for wall-clock timestamps

// Direct-form II transposed biquad (2nd-order IIR section).
// Stores two past inputs and two past outputs; call tick() once per sample.
struct Biquad {
    double x1{0.0}, x2{0.0};   // delayed inputs:  x[n-1], x[n-2]
    double y1{0.0}, y2{0.0};   // delayed outputs: y[n-1], y[n-2]

    // y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
    double tick(double x0, double b0, double b1, double b2,
                double a1, double a2) noexcept {
        double y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        x2 = x1;  x1 = x0;
        y2 = y1;  y1 = y0;
        return y0;
    }
};

// Maintains inter-call filter state for 2nd-order Butterworth velocity estimation.
// One instance lives for the lifetime of the control loop; call update() once per tick.
struct StateEstimator {
    StateEstimator() = default;   // zero-initialises all filter state

    // Reads both encoders, computes Butterworth-filtered velocities using actual
    // elapsed dt, and returns a fully populated StatePacket ready for transmission.
    StatePacket update(const EncoderState& enc_carriage,
                       const EncoderState& enc_pendulum);

private:
    long long prev_c1{0}, prev_c2{0};          // encoder counts from the previous tick
    Biquad    bq_xdot{};                        // 2nd-order LP filter for carriage velocity
    Biquad    bq_thdot{};                       // 2nd-order LP filter for pendulum angular velocity
    std::chrono::steady_clock::time_point last_time{};   // wall-clock time of the previous update call
    bool      first_call{true};                // true until the first update() completes; uses nominal DT on first tick
};
