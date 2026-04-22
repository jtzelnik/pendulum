#pragma once                     // guard against multiple inclusion
#include "../common/protocol.h"  // MotorCommand (wire struct shared with RL client)
#include "state.h"               // StateEstimator, EncoderState
#include <atomic>                // std::atomic<bool> for the shared stop flag

// 50 Hz closed-loop interface between the hardware and the client PC.
//   - Publishes a StatePacket on tcp:5555 (ZMQ_PUB) every tick.
//   - Pulls MotorCommands from tcp:5556 (ZMQ_PULL) each tick; uses the latest.
//   - Enforces limit-sensor safety on every received command before applying it.
//   - Blocks until the done flag is set, then shuts down sockets cleanly.
void lli_loop(EncoderState& enc_carriage, EncoderState& enc_pendulum,
              std::atomic<bool>& done);
