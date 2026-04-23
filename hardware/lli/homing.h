#pragma once
#include "pendulum_hw.h"   // EncoderState, pin constants, set_motor, HOMING_DUTY (defined in homing.cpp)
#include <atomic>          // std::atomic<bool> for the shared stop flag

// ── Full homing sequence ──────────────────────────────────────────────────────
// Performs a complete physical calibration of both axes from scratch.
// Called once at startup (lli_main.cpp) and every 10 episodes to correct drift.
//
// Steps:
//   1. Back off if the carriage is already touching a limit sensor.
//   2. Drive left to find the near limit; zero the carriage encoder there.
//   3. Drive right to find the far limit; measure the total rail count range.
//   4. Drive left to the midpoint (centre); re-zero the carriage encoder (x = 0).
//   5. Wait for the pendulum to hang still (< 2 count variation over 10 s).
//   6. Zero the pendulum encoder (theta = 0 = hanging straight down).
//
// Returns true on success.
// Returns false if the done flag is set during any phase — the caller should
// then stop the motor and exit rather than starting the control loop.
bool homing(EncoderState& enc_carriage, EncoderState& enc_pendulum,
            std::atomic<bool>& done);

// ── Fast centre-only re-home ──────────────────────────────────────────────────
// A quicker re-home used between most RL episodes to save time.
// Instead of re-scanning the rail limits, it drives directly back to encoder
// zero (the centre reference established by the last full homing), then waits
// for the pendulum to settle.
//
// Prerequisites: a full homing() must have been run at least once so that
// the encoder zero reference is valid. If the reference drifts significantly
// over many episodes, the next full homing will correct it.
//
// Returns true on success, false if interrupted by the done flag.
bool homing_center_only(EncoderState& enc_carriage, EncoderState& enc_pendulum,
                        std::atomic<bool>& done);
