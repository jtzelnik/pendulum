#include "pendulum_hw.h"   // pin constants, EncoderState, QUAD_TABLE, set_motor
#include "homing.h"        // homing() — establishes known zero reference before control begins
#include "lli_loop.h"      // lli_loop(), MotorCommand — the 50 Hz ZeroMQ control loop
#include <iostream>        // std::cout, std::cerr
#include <atomic>          // std::atomic for done flag and prox sensor cache
#include <thread>          // std::thread for ENTER-key listener
#include <csignal>         // signal(), SIGINT

std::atomic<bool> done{false};       // shared stop flag; set by SIGINT or ENTER key thread
std::atomic<int>  prox_near_val{-1}; // latest near-sensor level, cached by interrupt callback
std::atomic<int>  prox_far_val{-1};  // latest far-sensor  level, cached by interrupt callback

// Called by pigpio on any edge on ENC1_A, ENC1_B, ENC2_A, or ENC2_B.
// Decodes the quadrature transition and atomically updates the encoder count.
void encoder_cb(int gpio, int level, uint32_t /*tick*/, void* ud) {
    if (level == PI_TIMEOUT) return;                          // ignore pigpio watchdog timeout pseudo-events
    EncoderState* e = static_cast<EncoderState*>(ud);         // recover per-encoder state from user-data pointer
    if      (gpio == (int)e->pin_a) e->a = level;             // update cached A-channel level
    else if (gpio == (int)e->pin_b) e->b = level;             // update cached B-channel level
    else return;                                              // edge on unexpected pin — ignore
    int curr = (e->a << 1) | e->b;                            // pack A and B into 2-bit state index
    if (curr == e->prev) return;                              // no state change — spurious callback
    if ((e->prev ^ curr) == 3) { e->prev = curr; return; }   // both bits flipped simultaneously — illegal, discard
    int delta = QUAD_TABLE[(e->prev << 2) | curr];            // look up direction from previous→current transition
    if (delta) e->count.fetch_add(delta);                     // atomically update the signed count
    e->prev = curr;                                           // store current state for the next edge
}

// Called by pigpio on any edge on PROX_NEAR.
// Caches the level for display or diagnostics; safety logic reads GPIO directly.
void prox_near_cb(int /*gpio*/, int level, uint32_t /*tick*/, void* /*ud*/) {
    if (level != PI_TIMEOUT) prox_near_val.store(level);   // update cached near-sensor state
}

// Called by pigpio on any edge on PROX_FAR.
// Caches the level for display or diagnostics; safety logic reads GPIO directly.
void prox_far_cb(int /*gpio*/, int level, uint32_t /*tick*/, void* /*ud*/) {
    if (level != PI_TIMEOUT) prox_far_val.store(level);    // update cached far-sensor state
}

// Signal handler for SIGINT (Ctrl+C).
// Sets the shared done flag so homing and lli_loop exit cleanly.
void sigint_handler(int) { done = true; }

int main() {
    if (gpioInitialise() < 0) {                               // initialise pigpio; returns < 0 on failure
        std::cerr << "pigpio init failed - run with sudo\n";  // pigpio requires root for hardware access
        return 1;                                             // exit with error code
    }

    signal(SIGINT, sigint_handler);   // register Ctrl+C handler for clean shutdown

    EncoderState enc1(ENC1_A, ENC1_B);   // carriage encoder state
    EncoderState enc2(ENC2_A, ENC2_B);   // pendulum encoder state

    gpioSetMode(ENC1_A,    PI_INPUT);  gpioSetPullUpDown(ENC1_A,    PI_PUD_OFF);   // encoder input, external driver — no internal pull
    gpioSetMode(ENC1_B,    PI_INPUT);  gpioSetPullUpDown(ENC1_B,    PI_PUD_OFF);   // encoder input, external driver — no internal pull
    gpioSetMode(ENC2_A,    PI_INPUT);  gpioSetPullUpDown(ENC2_A,    PI_PUD_OFF);   // encoder input, external driver — no internal pull
    gpioSetMode(ENC2_B,    PI_INPUT);  gpioSetPullUpDown(ENC2_B,    PI_PUD_OFF);   // encoder input, external driver — no internal pull
    gpioSetMode(PROX_NEAR, PI_INPUT);  gpioSetPullUpDown(PROX_NEAR, PI_PUD_OFF);   // proximity sensor, external pull from sensor module
    gpioSetMode(PROX_FAR,  PI_INPUT);  gpioSetPullUpDown(PROX_FAR,  PI_PUD_OFF);   // proximity sensor, external pull from sensor module
    gpioSetMode(MOTOR_R,   PI_OUTPUT);   // R_PWM output to IBT2 motor driver
    gpioSetMode(MOTOR_L,   PI_OUTPUT);   // L_PWM output to IBT2 motor driver

    gpioGlitchFilter(PROX_NEAR, 1000);   // reject near-sensor pulses shorter than 1000 µs
    gpioGlitchFilter(PROX_FAR,  1000);   // reject far-sensor  pulses shorter than 1000 µs

    set_motor(0, 0);   // ensure motor is off before reading initial encoder state

    enc1.a = gpioRead(ENC1_A); enc1.b = gpioRead(ENC1_B);   // snapshot encoder 1 levels before enabling callbacks
    enc1.prev = (enc1.a << 1) | enc1.b;                      // build initial 2-bit state for correct first-edge decode
    enc2.a = gpioRead(ENC2_A); enc2.b = gpioRead(ENC2_B);   // snapshot encoder 2 levels before enabling callbacks
    enc2.prev = (enc2.a << 1) | enc2.b;                      // build initial 2-bit state for correct first-edge decode

    prox_near_val.store(gpioRead(PROX_NEAR));   // seed near-sensor cache before callback is registered
    prox_far_val.store(gpioRead(PROX_FAR));     // seed far-sensor  cache before callback is registered

    gpioSetAlertFuncEx(ENC1_A,    encoder_cb,   &enc1);     // edge callback for carriage encoder A
    gpioSetAlertFuncEx(ENC1_B,    encoder_cb,   &enc1);     // edge callback for carriage encoder B
    gpioSetAlertFuncEx(ENC2_A,    encoder_cb,   &enc2);     // edge callback for pendulum encoder A
    gpioSetAlertFuncEx(ENC2_B,    encoder_cb,   &enc2);     // edge callback for pendulum encoder B
    gpioSetAlertFuncEx(PROX_NEAR, prox_near_cb, nullptr);   // edge callback for near proximity sensor
    gpioSetAlertFuncEx(PROX_FAR,  prox_far_cb,  nullptr);   // edge callback for far  proximity sensor

    std::thread input_thread([]() { std::cin.get(); done = true; });   // sets done when operator presses ENTER

    if (!homing(enc1, enc2, done)) {   // run full homing sequence; returns false if interrupted
        set_motor(0, 0);               // safety: ensure motor off before exiting
        input_thread.join();           // wait for input thread to finish
        gpioTerminate();               // release pigpio resources
        return 1;                      // non-zero exit signals homing failure
    }

    lli_loop(enc1, enc2, done);   // hand off to 50 Hz ZeroMQ control loop; blocks until done

    set_motor(0, 0);        // ensure motor off after loop exits
    input_thread.join();    // wait for input thread to finish
    gpioTerminate();        // shut down pigpio and release all GPIO resources
    return 0;               // clean exit
}
