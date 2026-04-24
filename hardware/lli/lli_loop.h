#pragma once
#include "../common/protocol.h"  // MotorCommand — the command struct received from the RL client
#include "state.h"               // StateEstimator, EncoderState — used inside lli_loop
#include <atomic>                // std::atomic<bool> — shared stop flag polled each tick

// ── lli_loop ──────────────────────────────────────────────────────────────────
// The main 20 Hz real-time control loop. Call this after startup homing is done.
//
// Every 50 ms it:
//   1. Reads both encoders and publishes a StatePacket over ZeroMQ (port 5555).
//   2. Checks limit sensors and angular velocity for episode-ending conditions.
//   3. If a terminal was detected: stops the motor and runs a re-homing sequence.
//   4. Otherwise: drains the incoming command queue (port 5556) and applies the
//      most recent MotorCommand with safety overrides for the limit sensors.
//
// The function blocks until the done flag is set (Ctrl+C or ENTER), then
// closes all ZeroMQ sockets before returning.
//
// Parameters:
//   enc_carriage — shared encoder state for the linear (carriage) axis
//   enc_pendulum — shared encoder state for the angular (pendulum) axis
//   done         — set to true externally to request shutdown
void lli_loop(EncoderState& enc_carriage, EncoderState& enc_pendulum,
              std::atomic<bool>& done);
