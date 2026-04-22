#include "homing.h"    // function declaration, EncoderState, pin constants, set_motor
#include <iostream>    // std::cout, std::cerr for progress messages
#include <iomanip>     // std::setw for fixed-width countdown formatting
#include <thread>      // std::this_thread::sleep_for
#include <chrono>      // std::chrono::milliseconds, seconds
#include <deque>       // std::deque for the rolling stability window
#include <algorithm>   // std::min_element, std::max_element

static constexpr unsigned HOMING_DUTY  = 128;   // PWM duty used during homing (~50% of 255)
static constexpr int      STABLE_WINDOW = 10;   // seconds of pendulum quiet required to exit settle wait
static constexpr int      STABLE_TICKS  = 2;    // maximum tick range over the window to consider stable

// Blocks the calling thread for the given number of milliseconds.
static void pause_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));   // sleep for requested duration
}

// Full homing sequence. See homing.h for the step-by-step description.
// All motor motion uses HOMING_DUTY (~50%) for controlled, slow travel.
// Every blocking loop polls the done flag so SIGINT or ENTER aborts cleanly.
bool homing(EncoderState& enc_carriage, EncoderState& enc_pendulum,
            std::atomic<bool>& done)
{
    std::cout << "[homing] starting\n";   // notify operator that homing has begun

    // ── Step 1: back off if already at near limit ────────────────────────────
    if (gpioRead(PROX_NEAR) != 0) {                             // sensor reads non-zero → carriage is touching near stop
        std::cout << "[homing] already at near limit — backing off\n";
        set_motor(0, HOMING_DUTY);                              // drive right to clear the sensor
        while (gpioRead(PROX_NEAR) != 0 && !done) pause_ms(5); // wait until sensor clears or abort requested
        set_motor(0, 0);                                        // stop motor once clear
        pause_ms(300);                                          // let carriage decelerate fully before next move
    }
    if (done) return false;   // abort if stop flag was set during back-off

    // ── Step 2: find near limit ──────────────────────────────────────────────
    std::cout << "[homing] seeking near limit...\n";
    set_motor(HOMING_DUTY, 0);                              // drive left toward near stop
    while (gpioRead(PROX_NEAR) == 0 && !done) pause_ms(5); // wait until sensor triggers or abort requested
    set_motor(0, 0);                                        // stop motor on trigger
    if (done) return false;                                 // abort if stop flag set
    enc_carriage.count.store(0);                            // define near limit as carriage count zero
    std::cout << "[homing] near limit found — carriage count zeroed\n";
    pause_ms(300);                                          // settle before driving in the opposite direction

    // ── Step 3: find far limit, measure full range ───────────────────────────
    std::cout << "[homing] seeking far limit...\n";
    set_motor(0, HOMING_DUTY);                             // drive right toward far stop
    while (gpioRead(PROX_FAR) == 0 && !done) pause_ms(5); // wait until far sensor triggers or abort requested
    set_motor(0, 0);                                       // stop motor on trigger
    if (done) return false;                                // abort if stop flag set
    const long long total_counts = enc_carriage.count.load();   // total encoder counts across the full rail
    if (total_counts == 0) {                                     // zero range means encoder did not move — hardware fault
        std::cerr << "[homing] ERROR: encoder reported zero range — check wiring\n";
        return false;                                            // cannot home without a valid range
    }
    std::cout << "[homing] far limit found — range = " << total_counts
              << " counts (" << std::fixed << std::setprecision(4)
              << total_counts * METERS_PER_COUNT << " m)\n";   // print rail length in metres for verification
    pause_ms(300);                                              // settle before driving to centre

    // ── Step 4: drive to centre ──────────────────────────────────────────────
    const long long center = total_counts / 2;                 // target count is halfway between the two limits
    std::cout << "[homing] centering (target = " << center << " counts)...\n";
    set_motor(HOMING_DUTY, 0);                                 // drive left from far limit toward centre
    if (total_counts > 0) {                                    // positive range: count decreases as we go left
        while (enc_carriage.count.load() > center && !done) pause_ms(5);   // stop when count falls to centre
    } else {                                                   // negative range: count increases as we go left
        while (enc_carriage.count.load() < center && !done) pause_ms(5);   // stop when count rises to centre
    }
    set_motor(0, 0);                                           // stop motor at centre
    if (done) return false;                                    // abort if stop flag set
    enc_carriage.count.store(0);                               // redefine centre as x = 0
    std::cout << "[homing] carriage centered and zeroed\n";

    // ── Step 5: wait for pendulum to settle ──────────────────────────────────
    // Samples the pendulum encoder once per second into a rolling window.
    // Exits as soon as max - min across the last STABLE_WINDOW seconds is < STABLE_TICKS.
    std::cout << "[homing] waiting for pendulum to settle — press ENTER to abort\n";

    std::deque<long long> window;                              // rolling buffer of recent pendulum encoder readings
    for (int elapsed = 0; !done; ++elapsed) {                  // count up; no fixed timeout
        std::cout << "\r[homing] settling: " << std::setw(4) << elapsed
                  << "s elapsed...   " << std::flush;          // overwrite same line each second
        pause_ms(1000);                                        // wait one second between samples

        window.push_back(enc_pendulum.count.load());           // append latest pendulum count to window
        if ((int)window.size() > STABLE_WINDOW)               // keep window at fixed length
            window.pop_front();                                // drop oldest sample when full

        if ((int)window.size() == STABLE_WINDOW) {            // only evaluate stability once window is full
            long long lo = *std::min_element(window.begin(), window.end());   // minimum count in window
            long long hi = *std::max_element(window.begin(), window.end());   // maximum count in window
            if (hi - lo < STABLE_TICKS) break;                // range is below threshold — pendulum is still
        }
    }
    if (done) return false;                                    // abort if stop flag set during settle wait
    std::cout << "\n[homing] pendulum settled\n";

    // ── Step 6: zero pendulum encoder ────────────────────────────────────────
    enc_pendulum.count.store(0);                               // define current hanging position as theta = 0
    std::cout << "[homing] pendulum encoder zeroed — homing complete\n\n";
    return true;                                               // homing succeeded; system ready for control
}
