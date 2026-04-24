#include "state.h"   // StatePacket, StateEstimator, Biquad, DT, VEL_B*/A*, METERS_PER_COUNT, RAD_PER_COUNT

// Called once per tick by lli_loop (and manual_drive).
// Reads the current encoder counts, computes position and velocity for both
// axes, runs velocity through the Butterworth filter, and returns a fully
// populated StatePacket ready to be sent over ZeroMQ to the RL client.
StatePacket StateEstimator::update(const EncoderState& enc_carriage,
                                   const EncoderState& enc_pendulum)
{
    // Capture the current wall-clock time at the top of this call.
    // We measure actual elapsed time rather than assuming exactly DT seconds because
    // the OS scheduler and pigpio can cause the loop to wake a few microseconds
    // early or late. Using real dt keeps velocity accurate rather than
    // accumulating a small systematic error on every tick.
    auto now = std::chrono::steady_clock::now();

    double dt = DT;   // fall back to the nominal tick period on the very first call
    if (!first_call)  // after the first call, last_time is valid and we can measure the real gap
        dt = std::chrono::duration<double>(now - last_time).count();   // returns elapsed seconds as a double
    first_call = false;
    last_time  = now;   // store for next tick's dt measurement

    // Read both encoder counts atomically. The counts are written by the pigpio
    // interrupt callback (running on pigpio's internal thread), so .load() is
    // needed to ensure we see a fully-written 64-bit value — not a half-updated one.
    long long c1 = enc_carriage.count.load();   // carriage position in encoder ticks
    long long c2 = enc_pendulum.count.load();   // pendulum angle in encoder ticks

    // Finite-difference velocity: how far did each axis move in the last dt seconds?
    //   raw_xdot  (m/s)    = (Δcounts) × (metres/count) / (seconds)
    //   raw_thdot (rad/s)  = (Δcounts) × (radians/count) / (seconds)
    // The result is "raw" because encoder quantisation produces spiky noise:
    // a small real velocity might show as 0, 0, 2, 0, 1 counts per tick rather
    // than a smooth signal. The Butterworth filter below smooths this out.
    double raw_xdot  = static_cast<double>(c1 - prev_c1) * METERS_PER_COUNT / dt;
    double raw_thdot = static_cast<double>(c2 - prev_c2) * RAD_PER_COUNT    / dt;

    prev_c1 = c1;   // save for next tick's difference
    prev_c2 = c2;

    StatePacket pkt;

    // Timestamp in microseconds since the steady_clock epoch. The client can use
    // consecutive timestamps to verify the actual inter-packet interval.
    pkt.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           now.time_since_epoch()).count();

    // Convert raw counts to physical units.
    // After homing: enc_carriage.count == 0 at rail centre, so pkt.x == 0 there.
    //               enc_pendulum.count == 0 hanging straight down, so pkt.theta == 0 there.
    // Positive x = right of centre; positive theta = displaced from hanging (direction depends on mounting).
    pkt.x     = static_cast<double>(c1) * METERS_PER_COUNT;
    pkt.theta = static_cast<double>(c2) * RAD_PER_COUNT;

    // Run raw velocities through the 2nd-order Butterworth filter (Fc=LOOP_HZ/4, Fs=LOOP_HZ).
    // The same five coefficients (VEL_B0/B1/B2/A1/A2) are used for both axes;
    // each has its own Biquad instance so their delay lines remain independent.
    pkt.x_dot     = bq_xdot.tick (raw_xdot,  VEL_B0, VEL_B1, VEL_B2, VEL_A1, VEL_A2);
    pkt.theta_dot = bq_thdot.tick(raw_thdot, VEL_B0, VEL_B1, VEL_B2, VEL_A1, VEL_A2);

    return pkt;
}
