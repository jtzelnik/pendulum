#include "state.h"   // StatePacket, StateEstimator, DT, ALPHA, METERS_PER_COUNT, RAD_PER_COUNT

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

    // EMA velocity filter: new_vel = α * raw_vel + (1-α) * prev_vel
    x_dot  = ALPHA * (static_cast<double>(c1 - prev_c1) * METERS_PER_COUNT / dt)   // raw carriage velocity this tick
           + (1.0 - ALPHA) * x_dot;                                                 // blend with previous filtered value
    th_dot = ALPHA * (static_cast<double>(c2 - prev_c2) * RAD_PER_COUNT    / dt)   // raw pendulum velocity this tick
           + (1.0 - ALPHA) * th_dot;                                                // blend with previous filtered value

    prev_c1 = c1;   // store counts for next tick's delta calculation
    prev_c2 = c2;   // store counts for next tick's delta calculation

    StatePacket pkt;                                                                                    // construct output packet
    pkt.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(                          // convert time_point to integer microseconds
                           now.time_since_epoch()).count();                                             // since clock epoch
    pkt.x         = static_cast<double>(c1) * METERS_PER_COUNT;   // carriage position in metres
    pkt.theta     = static_cast<double>(c2) * RAD_PER_COUNT;      // pendulum angle in radians
    pkt.x_dot     = x_dot;                                         // filtered carriage velocity
    pkt.theta_dot = th_dot;                                        // filtered pendulum angular velocity
    return pkt;                                                    // return completed packet to caller
}
