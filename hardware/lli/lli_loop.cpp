#include "lli_loop.h"   // lli_loop() declaration, MotorCommand, StateEstimator, set_motor, pin constants
#include "homing.h"     // homing() and homing_center_only() — called after each episode terminal
#include <zmq.h>        // ZeroMQ C API: context, sockets, send/recv
#include <iostream>     // std::cout for status messages
#include <cstdio>       // std::printf for the live state display line
#include <thread>       // std::this_thread::sleep_until — used for fixed-rate timing
#include <chrono>       // std::chrono::steady_clock, milliseconds — clock and duration types
#include <cmath>        // std::abs — for the angular-velocity safety check

// Maximum safe pendulum angular velocity. If |theta_dot| exceeds this the
// pendulum is spinning too fast for the RL model to recover, so we end the
// episode and re-home rather than letting it thrash indefinitely.
static constexpr double THETA_DOT_LIMIT = 14.0;   // rad/s

// ── lli_loop ─────────────────────────────────────────────────────────────────
// The real-time control loop (LOOP_HZ Hz, set at compile time). This is the
// core of the LLI — it runs continuously during training and ties the hardware
// to the RL client.
//
// What happens every tick (LOOP_PERIOD_MS ms):
//   1. Sleep until the next scheduled tick time.
//   2. Read both encoders → compute state {x, ẋ, θ, θ̇}.
//   3. Check safety conditions (limit sensors, angular velocity).
//   4. Publish the StatePacket to the RL client over ZeroMQ PUB socket.
//   5. If a terminal condition was detected, stop the motor and re-home.
//   6. Otherwise, drain incoming MotorCommands from the ZeroMQ PULL socket
//      (keeping only the most recent) and apply the motor command with safety overrides.
//
// ZeroMQ socket roles:
//   PUB (port 5555) — broadcasts state to any subscribed client. The LLI
//     sends one packet per tick; if nobody is listening the packet is dropped.
//   PULL (port 5556) — receives motor commands pushed by the client.
//     HWM = 1 keeps only the most recent command; stale commands are discarded.
//
// Parameters:
//   enc_carriage — carriage encoder state (shared with encoder interrupt callback)
//   enc_pendulum — pendulum encoder state (shared with encoder interrupt callback)
//   done         — shared stop flag; loop exits when set to true
void lli_loop(EncoderState& enc_carriage, EncoderState& enc_pendulum,
              std::atomic<bool>& done)
{
    // ── ZeroMQ setup ─────────────────────────────────────────────────────────
    // A ZeroMQ "context" manages the background I/O threads. Create one per process.
    void* ctx  = zmq_ctx_new();

    // PUB socket: one sender, any number of receivers (subscribers).
    // The LLI binds (acts as server); the RL client connects (acts as client).
    void* pub  = zmq_socket(ctx, ZMQ_PUB);
    zmq_bind(pub, "tcp://*:5555");   // listen on all network interfaces, port 5555

    // PULL socket: receives motor commands pushed by the RL client.
    // High-water mark of 1 means at most one command can queue up inside ZeroMQ.
    // If the client sends commands faster than we read them, extras are dropped.
    // This prevents a backlog of stale commands from accumulating and being applied
    // several ticks after they were generated.
    void* pull = zmq_socket(ctx, ZMQ_PULL);
    int hwm = 1;
    zmq_setsockopt(pull, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    zmq_bind(pull, "tcp://*:5556");

    // Linger = 0: when we close the sockets, discard any unsent messages immediately
    // instead of waiting for them to drain (which could stall shutdown).
    int linger = 0;
    zmq_setsockopt(pub,  ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(pull, ZMQ_LINGER, &linger, sizeof(linger));

    // ── Loop state ───────────────────────────────────────────────────────────
    StateEstimator estimator;       // owns the Butterworth filter state; reset between episodes
    MotorCommand   cmd{0, 0, 0};    // the most recently received command; starts as "coast"
    int      episode_count = 0;     // tracks how many episodes have ended; used to schedule full vs fast re-home
    uint64_t tick_count    = 0;     // total ticks since loop start; used to print status every half second

    // Fixed-rate timing: we compute an absolute "next tick" time and sleep until it.
    // This is more accurate than sleeping for a fixed duration because it automatically
    // compensates for any time spent doing work inside the loop.
    using clock = std::chrono::steady_clock;
    auto next_tick = clock::now();

    std::cout << "[lli] running at " << LOOP_HZ << " Hz\n"
              << "[lli] PUB StatePacket   -> tcp:5555\n"
              << "[lli] PULL MotorCommand <- tcp:5556\n";

    while (!done) {
        // Advance the target time by one tick period and sleep until we reach it.
        next_tick += std::chrono::milliseconds(LOOP_PERIOD_MS);
        std::this_thread::sleep_until(next_tick);

        // ── State estimation ──────────────────────────────────────────────────
        // Read encoders and compute the full state vector. episode_status is
        // filled in below; start it at 0 (running normally).
        StatePacket pkt = estimator.update(enc_carriage, enc_pendulum);
        pkt.episode_status = 0;

        // ── Safety / terminal condition detection ─────────────────────────────
        // Read limit sensors directly from GPIO each tick (not from the cached
        // interrupt values) so we always have the very latest reading.
        bool at_near  = (gpioRead(PROX_NEAR) != 0);                      // carriage touching the near-stop limit
        bool at_far   = (gpioRead(PROX_FAR)  != 0);                      // carriage touching the far-stop limit
        bool over_vel = (std::abs(pkt.theta_dot) > THETA_DOT_LIMIT);     // pendulum spinning too fast

        // Limit hits take priority over angular velocity (both can be true simultaneously).
        if (at_near || at_far) pkt.episode_status = 1;
        else if (over_vel)     pkt.episode_status = 2;

        // ── Publish state ─────────────────────────────────────────────────────
        // Send the packet NOW, before homing, so the client sees the terminal
        // episode_status (1 or 2) and knows why the episode ended.
        // ZMQ_NOBLOCK: if no subscriber is connected the packet is silently dropped.
        zmq_send(pub, &pkt, sizeof(pkt), ZMQ_NOBLOCK);

        // ── Console display (every LOOP_HZ/2 ticks = half a second) ─────────
        if (++tick_count % (LOOP_HZ / 2) == 0) {
            std::printf("\r[lli]  x=%+6.3f m  x_dot=%+6.3f m/s  θ=%+6.3f rad  θ_dot=%+7.3f rad/s   ",
                        pkt.x, pkt.x_dot, pkt.theta, pkt.theta_dot);
            std::fflush(stdout);
        }

        // ── Handle episode terminal ───────────────────────────────────────────
        if (pkt.episode_status != 0) {
            set_motor(0, 0);      // stop the motor immediately
            cmd = {0, 0, 0};      // reset stored command so stale duty doesn't persist into the next episode
            ++episode_count;

            // Alternate between a fast centre-only re-home (most episodes) and
            // a full rail-scan re-home every 10 episodes to correct any drift in
            // the encoder zero reference that accumulates over time.
            bool full = (episode_count % 10 == 0);
            std::cout << "[lli] episode ended (status=" << (int)pkt.episode_status
                      << ", ep=" << episode_count
                      << ") — " << (full ? "full homing" : "center re-home") << "\n";

            bool ok = full ? homing(enc_carriage, enc_pendulum, done)
                           : homing_center_only(enc_carriage, enc_pendulum, done);
            if (!ok) break;   // homing was interrupted — exit the main loop

            // Double-flush the command queue after auto-homing.
            //
            // Why double-flush?  ZeroMQ's pipeline has two-message capacity:
            //   Slot 1 — the PULL socket's receive buffer (RCVHWM=1).
            //   Slot 2 — the TCP receive buffer on the Pi side.
            // When Slot 1 is full ZMQ stops draining from TCP, so Slot 2 fills.
            // The first flush empties Slot 1; that signals ZMQ to pull the TCP
            // message into Slot 1.  A short sleep lets the TCP delivery complete;
            // the second flush removes it. (This delay is a ZMQ pipeline constant,
            // not a function of LOOP_PERIOD_MS.)
            //
            // Without this a stale request_home that the client sent just before
            // detecting that homing was underway could survive the first flush
            // (sitting in TCP) and trigger an unwanted re-home on the next tick.
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }

            cmd = {0, 0, 0};
            estimator = StateEstimator{};   // reset filter history; velocities carry over otherwise and corrupt the first readings
            next_tick = clock::now();       // reset tick baseline so the first post-home tick isn't overdue
            std::cout << "[lli] re-homed — resuming control\n";
            continue;   // skip command processing for this iteration
        }

        // ── Receive latest command from PC ────────────────────────────────────
        // Drain all queued commands and keep only the last (most recent) one.
        // There could be more than one if the client sent faster than we read —
        // applying only the freshest avoids running an action that is already stale.
        MotorCommand incoming;
        while (zmq_recv(pull, &incoming, sizeof(incoming), ZMQ_NOBLOCK)
               == static_cast<int>(sizeof(incoming)))
            cmd = incoming;   // keep overwriting until the queue is empty

        // ── PC-requested re-home ──────────────────────────────────────────────
        // The client sets request_home=1 when a max-steps episode ends (status=0),
        // meaning the LLI did not auto-home and the client needs us to do it now.
        if (cmd.request_home) {
            set_motor(0, 0);
            cmd = {0, 0, 0};
            ++episode_count;
            bool full = (episode_count % 10 == 0);
            std::cout << "[lli] PC requested re-home (ep=" << episode_count
                      << ") — " << (full ? "full homing" : "center re-home") << "\n";

            // Publish a HOMING_STARTED acknowledgement (episode_status = 3) so the
            // client can confirm the request was received and homing is beginning.
            // Without this, the client would have to infer homing from several ticks of
            // silence, which is less reliable.
            {
                StatePacket ack = estimator.update(enc_carriage, enc_pendulum);
                ack.episode_status = 3;   // 3 = HOMING_STARTED
                zmq_send(pub, &ack, sizeof(ack), ZMQ_NOBLOCK);
            }

            bool ok = full ? homing(enc_carriage, enc_pendulum, done)
                           : homing_center_only(enc_carriage, enc_pendulum, done);
            if (!ok) break;

            // Double-flush the command queue with a short gap between flushes.
            // ZeroMQ's pipeline has two-message capacity: one in the PULL socket's
            // receive buffer (RCVHWM=1) and one held in the TCP send buffer on the
            // client side. The first flush drains the PULL buffer, which signals
            // ZMQ to accept the queued TCP message. The sleep lets that TCP delivery
            // complete, then the second flush removes it. Without the second flush
            // a stale request_home could trigger a second homing. (This delay is a
            // ZMQ pipeline constant, not a function of LOOP_PERIOD_MS.)
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            { MotorCommand tmp; while (zmq_recv(pull, &tmp, sizeof(tmp), ZMQ_NOBLOCK) > 0) {} }

            cmd = {0, 0, 0};
            estimator = StateEstimator{};
            next_tick = clock::now();
            std::cout << "[lli] re-homed — resuming control\n";
            continue;
        }

        // ── Safety overrides before applying motor command ────────────────────
        // Even if the client sends a non-zero duty, override it to zero if:
        //   • the client requested an emergency stop (estop flag), or
        //   • the command would drive the carriage further into an already-triggered limit.
        // This prevents the motor from pushing against the physical stop.
        if (cmd.estop
            || (at_near && cmd.duty < 0)   // driving left while already at the near limit
            || (at_far  && cmd.duty > 0))  // driving right while already at the far limit
            cmd.duty = 0;                  // override to coast

        // ── Apply motor command ───────────────────────────────────────────────
        // Positive duty = right, negative = left, zero = coast.
        // The IBT2 driver takes separate R_PWM and L_PWM signals, so we split
        // the signed duty into one active channel and one zero channel.
        if (cmd.duty > 0)
            set_motor(0, static_cast<unsigned>(cmd.duty));        // right
        else if (cmd.duty < 0)
            set_motor(static_cast<unsigned>(-cmd.duty), 0);       // left (negate to get positive magnitude)
        else
            set_motor(0, 0);                                       // coast
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    set_motor(0, 0);       // motor off before tearing down sockets
    zmq_close(pub);        // release PUB socket
    zmq_close(pull);       // release PULL socket
    zmq_ctx_destroy(ctx);  // destroy context; blocks until all sockets are fully closed
}
