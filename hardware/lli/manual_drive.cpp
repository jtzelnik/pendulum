#include "pendulum_hw.h"   // pin constants, EncoderState, QUAD_TABLE, set_motor
#include "homing.h"        // homing() — establishes known zero reference before driving
#include "state.h"         // StatePacket, StateEstimator
#include <iostream>        // std::cout, std::cerr for display output
#include <atomic>          // std::atomic for done and prox cache flags
#include <chrono>          // steady_clock for fixed-rate loop timing
#include <thread>          // std::thread for ENTER-key listener, sleep_until
#include <iomanip>         // std::setw, std::setprecision for formatted table output
#include <csignal>         // signal(), SIGINT for Ctrl+C handling

std::atomic<bool> done{false};              // shared stop flag; set by SIGINT handler or ENTER key thread
std::atomic<int>  prox_near_val{-1};        // latest near-sensor level cached by interrupt callback
std::atomic<int>  prox_far_val{-1};         // latest far-sensor  level cached by interrupt callback

// Called by pigpio on any edge on ENC1_A, ENC1_B, ENC2_A, or ENC2_B.
// Decodes the quadrature transition and atomically updates the encoder count.
void encoder_cb(int gpio, int level, uint32_t /*tick*/, void* ud) {
    if (level == PI_TIMEOUT) return;                          // ignore pigpio watchdog timeout events
    EncoderState* e = static_cast<EncoderState*>(ud);         // recover per-encoder state from user-data pointer
    if      (gpio == (int)e->pin_a) e->a = level;             // update cached A-channel level
    else if (gpio == (int)e->pin_b) e->b = level;             // update cached B-channel level
    else return;                                              // edge on unexpected pin — ignore
    int curr = (e->a << 1) | e->b;                            // pack A and B into a 2-bit state index
    if (curr == e->prev) return;                              // no state change — spurious callback
    if ((e->prev ^ curr) == 3) { e->prev = curr; return; }   // both bits flipped simultaneously — illegal transition, discard
    int delta = QUAD_TABLE[(e->prev << 2) | curr];            // look up direction from previous→current state transition
    if (delta) e->count.fetch_add(delta);                     // atomically increment or decrement the count
    e->prev = curr;                                           // store current state for next edge
}

// Called by pigpio on any edge on PROX_NEAR.
// Caches the raw level so the main loop can read it without a gpioRead() call.
void prox_near_cb(int /*gpio*/, int level, uint32_t /*tick*/, void* /*ud*/) {
    if (level != PI_TIMEOUT) prox_near_val.store(level);   // update cached near-sensor state
}

// Called by pigpio on any edge on PROX_FAR.
// Caches the raw level so the main loop can read it without a gpioRead() call.
void prox_far_cb(int /*gpio*/, int level, uint32_t /*tick*/, void* /*ud*/) {
    if (level != PI_TIMEOUT) prox_far_val.store(level);    // update cached far-sensor state
}

// Signal handler for SIGINT (Ctrl+C).
// Sets the shared done flag so all loops exit cleanly on the next iteration.
void sigint_handler(int) { done = true; }

int main() {
    if (gpioInitialise() < 0) {                              // initialise pigpio; returns < 0 on failure
        std::cerr << "pigpio init failed - run with sudo\n"; // pigpio requires root for hardware access
        return 1;                                            // exit with error code
    }

    signal(SIGINT, sigint_handler);   // register Ctrl+C handler so the process shuts down gracefully

    EncoderState enc1(ENC1_A, ENC1_B);   // carriage encoder state — tracks linear position
    EncoderState enc2(ENC2_A, ENC2_B);   // pendulum encoder state — tracks angular position

    gpioSetMode(ENC1_A,    PI_INPUT);  gpioSetPullUpDown(ENC1_A,    PI_PUD_OFF);   // encoder lines: floating input, external differential driver
    gpioSetMode(ENC1_B,    PI_INPUT);  gpioSetPullUpDown(ENC1_B,    PI_PUD_OFF);   // encoder lines: floating input, external differential driver
    gpioSetMode(ENC2_A,    PI_INPUT);  gpioSetPullUpDown(ENC2_A,    PI_PUD_OFF);   // encoder lines: floating input, external differential driver
    gpioSetMode(ENC2_B,    PI_INPUT);  gpioSetPullUpDown(ENC2_B,    PI_PUD_OFF);   // encoder lines: floating input, external differential driver
    gpioSetMode(PROX_NEAR, PI_INPUT);  gpioSetPullUpDown(PROX_NEAR, PI_PUD_OFF);   // prox sensor: external pull provided by sensor module
    gpioSetMode(PROX_FAR,  PI_INPUT);  gpioSetPullUpDown(PROX_FAR,  PI_PUD_OFF);   // prox sensor: external pull provided by sensor module
    gpioSetMode(BTN_RIGHT, PI_INPUT);  gpioSetPullUpDown(BTN_RIGHT, PI_PUD_UP);    // button: internal pull-up, active-low when pressed
    gpioSetMode(BTN_LEFT,  PI_INPUT);  gpioSetPullUpDown(BTN_LEFT,  PI_PUD_UP);    // button: internal pull-up, active-low when pressed
    gpioSetMode(MOTOR_R,   PI_OUTPUT);   // R_PWM output to IBT2
    gpioSetMode(MOTOR_L,   PI_OUTPUT);   // L_PWM output to IBT2

    gpioGlitchFilter(PROX_NEAR, 1000);   // reject near-sensor pulses shorter than 1000 µs (noise rejection)
    gpioGlitchFilter(PROX_FAR,  1000);   // reject far-sensor  pulses shorter than 1000 µs (noise rejection)
    gpioGlitchFilter(BTN_RIGHT, 5000);   // debounce right button: ignore transitions shorter than 5000 µs
    gpioGlitchFilter(BTN_LEFT,  5000);   // debounce left  button: ignore transitions shorter than 5000 µs

    set_motor(0, 0);   // ensure motor is off before initialising encoder state

    enc1.a = gpioRead(ENC1_A); enc1.b = gpioRead(ENC1_B);     // read initial A and B levels for encoder 1
    enc1.prev = (enc1.a << 1) | enc1.b;                        // build initial 2-bit state so first edge is decoded correctly
    enc2.a = gpioRead(ENC2_A); enc2.b = gpioRead(ENC2_B);     // read initial A and B levels for encoder 2
    enc2.prev = (enc2.a << 1) | enc2.b;                        // build initial 2-bit state so first edge is decoded correctly

    prox_near_val.store(gpioRead(PROX_NEAR));   // initialise near-sensor cache before enabling callback
    prox_far_val.store(gpioRead(PROX_FAR));     // initialise far-sensor  cache before enabling callback

    gpioSetAlertFuncEx(ENC1_A,    encoder_cb,   &enc1);     // register edge callback for encoder 1 A-channel
    gpioSetAlertFuncEx(ENC1_B,    encoder_cb,   &enc1);     // register edge callback for encoder 1 B-channel
    gpioSetAlertFuncEx(ENC2_A,    encoder_cb,   &enc2);     // register edge callback for encoder 2 A-channel
    gpioSetAlertFuncEx(ENC2_B,    encoder_cb,   &enc2);     // register edge callback for encoder 2 B-channel
    gpioSetAlertFuncEx(PROX_NEAR, prox_near_cb, nullptr);   // register edge callback for near proximity sensor
    gpioSetAlertFuncEx(PROX_FAR,  prox_far_cb,  nullptr);   // register edge callback for far  proximity sensor

    std::thread input_thread([]() { std::cin.get(); done = true; });   // background thread: sets done when ENTER is pressed

    if (!homing(enc1, enc2, done)) {   // run homing; returns false if interrupted
        set_motor(0, 0);               // ensure motor off on early exit
        input_thread.join();           // wait for input thread to finish
        gpioTerminate();               // shut down pigpio and release GPIO
        return 1;                      // exit with error to signal incomplete homing
    }

    std::cout << "Manual drive at 50 Hz. Press ENTER to stop.\n\n";
    std::cout << std::fixed << std::setprecision(4);   // fixed-point notation, 4 decimal places for all floats
    std::cout << std::left                             // left-align all columns
        << std::setw(10) << "x(m)"
        << std::setw(12) << "x_dot(m/s)"
        << std::setw(12) << "theta(rad)"
        << std::setw(14) << "th_dot(r/s)"
        << std::setw(6)  << "near"
        << std::setw(6)  << "far"
        << std::setw(8)  << "motor"
        << std::setw(8)  << "status"
        << std::setw(10) << "dt(ms)"
        << '\n';                                       // print column header row

    StateEstimator estimator;   // owns Butterworth filter state across loop iterations
    int64_t prev_ts = 0;        // timestamp of previous packet, used to compute actual dt for display

    using clock = std::chrono::steady_clock;   // monotonic clock alias
    auto next_tick = clock::now();             // absolute time of the next scheduled tick

    while (!done) {                                              // run until done flag is set
        next_tick += std::chrono::milliseconds(20);              // advance tick target by 20 ms
        std::this_thread::sleep_until(next_tick);                // sleep until next tick boundary

        StatePacket pkt = estimator.update(enc1, enc2);          // compute current state from encoders
        pkt.episode_status = 0;                                  // manual drive does not use episode logic; always running

        double actual_dt_ms = prev_ts > 0                        // compute real elapsed time in milliseconds
            ? (pkt.timestamp_us - prev_ts) / 1000.0             // convert microsecond delta to milliseconds
            : DT * 1000.0;                                       // use nominal value on the very first tick
        prev_ts = pkt.timestamp_us;                              // store timestamp for next iteration

        bool btn_r  = (gpioRead(BTN_RIGHT) == 0);   // true when right button is pressed (active-low)
        bool btn_l  = (gpioRead(BTN_LEFT)  == 0);   // true when left  button is pressed (active-low)
        bool safe_r = (gpioRead(PROX_FAR)  == 0);   // true when far  sensor is not triggered (safe to drive right)
        bool safe_l = (gpioRead(PROX_NEAR) == 0);   // true when near sensor is not triggered (safe to drive left)

        const char* motor_state;                                              // string label for current motor state shown in table
        if (btn_r && !btn_l && safe_r) {                                      // right button only, and not at far limit
            set_motor(0, MOTOR_DUTY);                                         // drive right
            motor_state = "RIGHT";
        } else if (btn_l && !btn_r && safe_l) {                               // left button only, and not at near limit
            set_motor(MOTOR_DUTY, 0);                                         // drive left
            motor_state = "LEFT";
        } else {                                                              // no valid drive condition
            set_motor(0, 0);                                                  // coast
            motor_state = ((!safe_r && btn_r) || (!safe_l && btn_l))         // button pressed but sensor blocked
                          ? "LIMIT" : "COAST";                                // label accordingly
        }

        std::cout << std::left
            << std::setw(10) << pkt.x            // carriage position in metres
            << std::setw(12) << pkt.x_dot        // filtered carriage velocity
            << std::setw(12) << pkt.theta        // pendulum angle in radians
            << std::setw(14) << pkt.theta_dot    // filtered pendulum angular velocity
            << std::setw(6)  << prox_near_val.load()   // latest near-sensor level from interrupt cache
            << std::setw(6)  << prox_far_val.load()    // latest far-sensor  level from interrupt cache
            << std::setw(8)  << motor_state             // current motor drive state label
            << std::setw(8)  << (int)pkt.episode_status // episode status (always 0 in manual drive)
            << std::setw(10) << actual_dt_ms            // actual loop period in milliseconds
            << '\n';                                    // flush one row per tick
    }

    set_motor(0, 0);        // ensure motor is off before teardown
    input_thread.join();    // wait for input thread to exit
    gpioTerminate();        // shut down pigpio and release all GPIO resources
    return 0;               // clean exit
}
