#pragma once
#include "../common/protocol.h"  // StatePacket — the wire struct we populate and send over ZeroMQ
#include "pendulum_hw.h"         // EncoderState, VEL_B*/A* filter coefficients, METERS_PER_COUNT, RAD_PER_COUNT, DT
#include <chrono>                // std::chrono::steady_clock — monotonic clock for measuring actual tick duration

// ── Biquad (2nd-order IIR digital filter) ────────────────────────────────────
// A biquad is the standard building block for digital filters.
//
// "IIR" (Infinite Impulse Response) means the output depends on past outputs
// as well as past inputs. Feeding the output back in (via a1 and a2) lets a
// 2nd-order section do work that would take many more taps in a simpler filter.
//
// This implements the "Direct Form II Transposed" structure, which has good
// numerical precision for the coefficient ranges we use.
//
// State variables stored between calls:
//   x1, x2 — input from 1 tick ago, 2 ticks ago   (the "input memory")
//   y1, y2 — output from 1 tick ago, 2 ticks ago  (the "output feedback")
//
// Difference equation applied each tick:
//   y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
//
// The b0/b1/b2 coefficients weight how much the current and recent raw inputs
// contribute. The a1/a2 coefficients feed past filtered outputs back in, which
// is what gives the filter its frequency-selective "memory".
struct Biquad {
    double x1{0.0}, x2{0.0};   // input delay line: x[n-1] and x[n-2]
    double y1{0.0}, y2{0.0};   // output delay line: y[n-1] and y[n-2]

    // Apply one filter tick and return the filtered output.
    //   x0      — raw (unfiltered) input for the current tick
    //   b0,b1,b2 — feedforward (numerator) coefficients
    //   a1,a2    — feedback (denominator) coefficients; subtracted in the equation
    // noexcept: this function never throws, which lets the compiler optimise more aggressively.
    double tick(double x0, double b0, double b1, double b2,
                double a1, double a2) noexcept {
        double y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2;   // apply the difference equation
        x2 = x1;  x1 = x0;   // shift input history one step: x1 → x2, new input → x1
        y2 = y1;  y1 = y0;   // shift output history one step: y1 → y2, new output → y1
        return y0;
    }
};

// ── StateEstimator ───────────────────────────────────────────────────────────
// Converts raw encoder counts into the full physical state {x, ẋ, θ, θ̇}.
//
// One instance lives for the lifetime of the control loop. Calling update()
// exactly once per 20 Hz tick keeps the filter's internal delay-line state
// contiguous and correct. Resetting the instance (= StateEstimator{}) between
// episodes clears all filter memory so velocity estimates start fresh.
//
// Velocity estimation pipeline per tick:
//   1. Read encoder counts (atomic, thread-safe).
//   2. Subtract previous counts → raw position change (Δcounts).
//   3. Divide by actual elapsed time → raw velocity in physical units.
//   4. Pass through Biquad Butterworth filter → smooth velocity for the client.
struct StateEstimator {
    StateEstimator() = default;   // zero-initialises all fields; filter starts with no history

    // Read both encoders, compute filtered velocities, and return a StatePacket
    // ready to be sent over ZeroMQ. Must be called exactly once per control tick.
    StatePacket update(const EncoderState& enc_carriage,
                       const EncoderState& enc_pendulum);

private:
    long long prev_c1{0}, prev_c2{0};   // encoder counts stored from the previous tick, used to compute velocity by differencing
    Biquad    bq_xdot{};                 // filter instance for carriage velocity (x_dot)
    Biquad    bq_thdot{};                // filter instance for pendulum angular velocity (theta_dot)
    std::chrono::steady_clock::time_point last_time{};   // wall-clock time recorded at the end of the last update() call
    bool      first_call{true};          // suppresses dt measurement on the very first call (last_time not yet valid)
};
