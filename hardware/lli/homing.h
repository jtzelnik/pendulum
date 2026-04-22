#pragma once               // guard against multiple inclusion
#include "pendulum_hw.h"   // EncoderState, pin constants, set_motor
#include <atomic>          // std::atomic<bool> for the shared stop flag

// Runs the full homing sequence:
//   1. Back off if already touching a limit sensor.
//   2. Drive to near limit, zero carriage encoder.
//   3. Drive to far limit, measure total rail length.
//   4. Drive to centre, zero carriage encoder (x = 0 at centre).
//   5. Wait until pendulum is stationary (< 2 encoder ticks over 10 s).
//   6. Zero pendulum encoder (theta = 0 hanging down).
//
// Returns true on success. Returns false if the done flag is set during
// any phase (e.g. user pressed ENTER or SIGINT received).
bool homing(EncoderState& enc_carriage, EncoderState& enc_pendulum,
            std::atomic<bool>& done);

// Fast re-home used between most episodes: skips the limit scan and drives
// directly back to encoder zero (the centre established by the last full
// homing), then waits for the pendulum to settle and re-zeros it.
// Requires that a full homing has already been performed so the encoder
// reference is valid.  Backs off from either limit if triggered before driving.
bool homing_center_only(EncoderState& enc_carriage, EncoderState& enc_pendulum,
                        std::atomic<bool>& done);
