#include "lli_loop.h"   // MotorCommand, lli_loop declaration, StateEstimator, set_motor, pin constants
#include "homing.h"     // homing() — re-home after each episode terminal
#include <zmq.h>        // ZeroMQ C API — context, sockets, send/recv
#include <iostream>     // std::cout for startup messages
#include <cstdio>       // std::printf for the live state display line
#include <thread>       // std::this_thread::sleep_until
#include <chrono>       // std::chrono::steady_clock, milliseconds
#include <cmath>        // std::abs for angular velocity threshold check

static constexpr double THETA_DOT_LIMIT = 14.0;   // rad/s — exceeding this ends the episode immediately

// 50 Hz control loop. Publishes state to the client and applies motor commands
// received from the client, with hard safety overrides at the limit sensors.
// Detects episode-ending conditions (limit hit or angular velocity exceeded),
// flags them in the state packet, then automatically re-homes before resuming.
// Blocks until done is set, then tears down ZeroMQ sockets before returning.
void lli_loop(EncoderState& enc_carriage, EncoderState& enc_pendulum,
              std::atomic<bool>& done)
{
    // ── ZeroMQ context and socket setup ──────────────────────────────────────
    void* ctx  = zmq_ctx_new();                                        // create a ZeroMQ context (one per process)

    void* pub  = zmq_socket(ctx, ZMQ_PUB);                            // PUB socket — broadcasts state to all subscribers
    zmq_bind(pub, "tcp://*:5555");                                     // listen on all interfaces, port 5555

    void* pull = zmq_socket(ctx, ZMQ_PULL);                           // PULL socket — receives commands pushed by client
    int hwm = 1;                                                       // high-water mark: keep at most 1 message queued
    zmq_setsockopt(pull, ZMQ_RCVHWM, &hwm, sizeof(hwm));              // prevents stale commands accumulating between ticks
    zmq_bind(pull, "tcp://*:5556");                                    // listen on all interfaces, port 5556

    int linger = 0;                                                    // close sockets immediately on shutdown
    zmq_setsockopt(pub,  ZMQ_LINGER, &linger, sizeof(linger));        // discard unsent state messages on close
    zmq_setsockopt(pull, ZMQ_LINGER, &linger, sizeof(linger));        // discard unread command messages on close

    // ── Loop state ───────────────────────────────────────────────────────────
    StateEstimator estimator;      // owns the Butterworth filter state across ticks
    MotorCommand   cmd{0, 0, 0};   // last received command; initialised to coast
    int      episode_count = 0;    // re-homes completed; full homing every 10 episodes
    uint64_t tick_count    = 0;    // total ticks since loop start; used for periodic console print

    using clock = std::chrono::steady_clock;   // monotonic clock alias
    auto next_tick = clock::now();             // absolute time of the next scheduled tick

    std::cout << "[lli] running at 50 Hz\n"
              << "[lli] PUB StatePacket   -> tcp:5555\n"   // inform operator which ports are active
              << "[lli] PULL MotorCommand <- tcp:5556\n";

    while (!done) {                                                    // run until SIGINT or ENTER
        next_tick += std::chrono::milliseconds(20);                    // advance target time by one 20 ms period
        std::this_thread::sleep_until(next_tick);                      // sleep until the next tick boundary

        // ── State estimation ──────────────────────────────────────────────────
        StatePacket pkt = estimator.update(enc_carriage, enc_pendulum);   // compute {x, x_dot, theta, theta_dot, timestamp}
        pkt.episode_status = 0;                                           // default: episode is running normally

        // ── Episode termination detection ─────────────────────────────────────
        bool at_near  = (gpioRead(PROX_NEAR) != 0);                       // near sensor non-zero → carriage touching near stop
        bool at_far   = (gpioRead(PROX_FAR)  != 0);                       // far  sensor non-zero → carriage touching far  stop
        bool over_vel = (std::abs(pkt.theta_dot) > THETA_DOT_LIMIT);      // pendulum angular speed exceeded safe threshold

        if (at_near || at_far) pkt.episode_status = 1;                    // limit hit takes priority in status code
        else if (over_vel)     pkt.episode_status = 2;                    // angular velocity exceeded threshold

        // ── Publish state (including terminal flag if set) ────────────────────
        zmq_send(pub, &pkt, sizeof(pkt), ZMQ_NOBLOCK);   // send packet; client sees episode_status before re-homing begins

        // ── Periodic console state display (every 0.5 s = 25 ticks) ─────────
        if (++tick_count % 25 == 0) {
            std::printf("\r[lli]  x=%+6.3f m  x_dot=%+6.3f m/s  θ=%+6.3f rad  θ_dot=%+7.3f rad/s   ",
                        pkt.x, pkt.x_dot, pkt.theta, pkt.theta_dot);
            std::fflush(stdout);
        }

        // ── Handle episode termination ────────────────────────────────────────
        if (pkt.episode_status != 0) {                                    // a terminal condition was detected this tick
            set_motor(0, 0);                                              // immediately stop motor
            cmd = {0, 0, 0};                                              // reset stored command to coast so stale duty doesn't persist
            ++episode_count;
            bool full = (episode_count % 10 == 0);                        // full homing every 10 episodes
            std::cout << "[lli] episode ended (status=" << (int)pkt.episode_status
                      << ", ep=" << episode_count
                      << ") — " << (full ? "full homing" : "center re-home") << "\n";
            bool ok = full ? homing(enc_carriage, enc_pendulum, done)
                           : homing_center_only(enc_carriage, enc_pendulum, done);
            if (!ok) break;                                               // exit loop if interrupted during homing
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }   // flush commands that arrived during homing
            cmd = {0, 0, 0};
            estimator = StateEstimator{};                                 // reset filter state so velocities don't carry over from previous episode
            next_tick = clock::now();                                     // reset tick baseline so first post-home tick isn't overdue
            std::cout << "[lli] re-homed — resuming control\n";
            continue;                                                     // skip command processing for this iteration
        }

        // ── Receive latest command (drain queue, keep last) ───────────────────
        MotorCommand incoming;                                             // temporary storage for each received message
        while (zmq_recv(pull, &incoming, sizeof(incoming), ZMQ_NOBLOCK)   // non-blocking receive; returns -1 if queue empty
               == static_cast<int>(sizeof(incoming)))                     // only accept messages of exactly the expected size
            cmd = incoming;                                               // overwrite with each newer message; exits when queue empty

        // ── PC-requested re-home ─────────────────────────────────────────────
        if (cmd.request_home) {
            set_motor(0, 0);
            cmd = {0, 0, 0};
            ++episode_count;
            bool full = (episode_count % 10 == 0);
            std::cout << "[lli] PC requested re-home (ep=" << episode_count
                      << ") — " << (full ? "full homing" : "center re-home") << "\n";
            // Publish handshake before homing begins so client can confirm
            // initiation without relying on the 500 ms silence heuristic.
            {
                StatePacket ack = estimator.update(enc_carriage, enc_pendulum);
                ack.episode_status = 3;   // HOMING_STARTED
                zmq_send(pub, &ack, sizeof(ack), ZMQ_NOBLOCK);
            }
            bool ok = full ? homing(enc_carriage, enc_pendulum, done)
                           : homing_center_only(enc_carriage, enc_pendulum, done);
            if (!ok) break;
            // The ZMQ pipeline has two-message capacity: one in the PULL receive
            // buffer (RCVHWM=1) and one in the client's PUSH send buffer (SNDHWM=1).
            // The first flush drains the PULL buffer, which opens capacity so the
            // message held in the client's PUSH buffer is immediately delivered.
            // The 50 ms sleep lets that delivery complete, then the second flush
            // removes it — preventing a stale request_home from triggering a second
            // homing on the very next tick.
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }
            cmd = {0, 0, 0};
            estimator = StateEstimator{};
            next_tick = clock::now();
            std::cout << "[lli] re-homed — resuming control\n";
            continue;
        }

        // ── Safety overrides ──────────────────────────────────────────────────
        if (cmd.estop                                                     // client requested emergency stop
            || (at_near && cmd.duty < 0)                                  // command would drive into near limit
            || (at_far  && cmd.duty > 0))                                 // command would drive into far  limit
            cmd.duty = 0;                                                 // override: coast regardless of received duty

        // ── Apply motor command ───────────────────────────────────────────────
        if (cmd.duty > 0)                                                 // positive duty → drive right
            set_motor(0, static_cast<unsigned>(cmd.duty));                // R_PWM = 0, L_PWM = duty
        else if (cmd.duty < 0)                                            // negative duty → drive left
            set_motor(static_cast<unsigned>(-cmd.duty), 0);              // R_PWM = |duty|, L_PWM = 0
        else                                                              // zero duty → coast
            set_motor(0, 0);                                              // both channels off
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    set_motor(0, 0);          // ensure motor is off before tearing down sockets
    zmq_close(pub);           // close PUB socket and free its resources
    zmq_close(pull);          // close PULL socket and free its resources
    zmq_ctx_destroy(ctx);     // destroy ZeroMQ context; blocks until all sockets are closed
}
