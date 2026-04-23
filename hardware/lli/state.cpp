#include "state.h"   // StatePacket, StateEstimator, Biquad, DT, VEL_B*, VEL_A*, METERS_PER_COUNT, RAD_PER_COUNT

// Computes one tick of state estimation.
// Measures actual elapsed time since the last call so velocity is accurate
// even if the loop runs slightly early or late. Falls back to the nominal
// DT on the very first call when no previous timestamp exists.
StatePacket StateEstimator::update(const EncoderState& enc_carriage,
                                   const EncoderState& enc_pendulum)
{
    auto now = std::chrono::steady_clock::now();   // capture wall-clock time at start of this tick

    double dt = DT;                                // default to nominal period on first call
    if (!first_call)                               // after first call, measure real elapsed time
        dt = std::chrono::duration<double>(now - last_time).count();   // seconds between this tick and last
    first_call = false;                            // mark first call complete
    last_time  = now;                              // store timestamp for next tick's dt calculation

    long long c1 = enc_carriage.count.load();      // atomically read current carriage encoder count
    long long c2 = enc_pendulum.count.load();      // atomically read current pendulum encoder count

    // Raw finite-difference velocities for this tick
    double raw_xdot  = static_cast<double>(c1 - prev_c1) * METERS_PER_COUNT / dt;
    double raw_thdot = static_cast<double>(c2 - prev_c2) * RAD_PER_COUNT    / dt;

    prev_c1 = c1;
    prev_c2 = c2;

    StatePacket pkt;
    pkt.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           now.time_since_epoch()).count();
    pkt.x         = static_cast<double>(c1) * METERS_PER_COUNT;
    pkt.theta     = static_cast<double>(c2) * RAD_PER_COUNT;
    // 2nd-order Butterworth LP (Fc=20 Hz, Fs=50 Hz) applied to raw finite-difference velocity
    pkt.x_dot     = bq_xdot.tick (raw_xdot,  VEL_B0, VEL_B1, VEL_B2, VEL_A1, VEL_A2);
    pkt.theta_dot = bq_thdot.tick(raw_thdot, VEL_B0, VEL_B1, VEL_B2, VEL_A1, VEL_A2);
    return pkt;
}
