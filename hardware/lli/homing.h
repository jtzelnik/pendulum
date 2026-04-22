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
