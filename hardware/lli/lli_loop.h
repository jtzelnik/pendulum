#pragma once               // guard against multiple inclusion
#include "state.h"         // StatePacket, StateEstimator, EncoderState
#include <atomic>          // std::atomic<bool> for the shared stop flag
#include <cstdint>         // int32_t, uint8_t

// Command message sent from the client PC to the LLI each control tick.
// Wire layout is fixed — ZeroMQ receives this struct as raw bytes.
struct MotorCommand {
    int32_t  duty;    // signed PWM duty: +255 = full right, -255 = full left, 0 = coast
    uint8_t  estop;   // non-zero triggers immediate motor stop regardless of duty value
};

// 50 Hz closed-loop interface between the hardware and the client PC.
//   - Publishes a StatePacket on tcp:5555 (ZMQ_PUB) every tick.
//   - Pulls MotorCommands from tcp:5556 (ZMQ_PULL) each tick; uses the latest.
//   - Enforces limit-sensor safety on every received command before applying it.
//   - Blocks until the done flag is set, then shuts down sockets cleanly.
void lli_loop(EncoderState& enc_carriage, EncoderState& enc_pendulum,
              std::atomic<bool>& done);
