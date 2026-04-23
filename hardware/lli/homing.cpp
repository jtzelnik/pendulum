#include "homing.h"    // function declarations, EncoderState, pin constants, set_motor
#include <iostream>    // std::cout, std::cerr for progress messages
#include <iomanip>     // std::setw for fixed-width countdown formatting
#include <thread>      // std::this_thread::sleep_for — blocking waits between sensor checks
#include <chrono>      // std::chrono::milliseconds, seconds — duration types
#include <deque>       // std::deque — sliding window of pendulum encoder readings for stability check
#include <algorithm>   // std::min_element, std::max_element — range over the stability window

// PWM duty cycle used during homing. Lower than MOTOR_DUTY (230) so the
// carriage moves slowly and doesn't slam into the limit stops.
static constexpr unsigned HOMING_DUTY  = 128;   // ~50% power

// Stability detection parameters for the pendulum settle wait.
// The pendulum must show less than STABLE_TICKS variation in its encoder
// count over the last STABLE_WINDOW seconds before homing considers it still.
static constexpr int STABLE_WINDOW = 10;   // seconds of history to examine
static constexpr int STABLE_TICKS  = 2;    // max encoder-count range allowed over the window

// Helper: block the calling thread for the given number of milliseconds.
static void pause_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ── Full homing sequence ──────────────────────────────────────────────────────
// All motor motion uses HOMING_DUTY for slow, controlled travel.
// Every blocking loop polls done so SIGINT or ENTER aborts cleanly.
bool homing(EncoderState& enc_carriage, EncoderState& enc_pendulum,
            std::atomic<bool>& done)
{
    std::cout << "[homing] starting\n";

    // ── Step 1: back off if already at near limit ─────────────────────────────
    // The carriage may be resting against the near stop from a previous episode.
    // We must clear the sensor before trying to seek it in step 2, otherwise
    // the seek loop exits immediately and we never get an accurate count zero.
    if (gpioRead(PROX_NEAR) != 0) {
        std::cout << "[homing] already at near limit — backing off\n";
        set_motor(0, HOMING_DUTY);                              // drive right to clear
        while (gpioRead(PROX_NEAR) != 0 && !done) pause_ms(5); // wait until sensor clears
        set_motor(0, 0);
        pause_ms(300);   // let the carriage decelerate before the next move
    }
    if (done) return false;

    // ── Step 2: find near limit and zero carriage encoder ────────────────────
    std::cout << "[homing] seeking near limit...\n";
    set_motor(HOMING_DUTY, 0);                              // drive left toward near stop
    while (gpioRead(PROX_NEAR) == 0 && !done) pause_ms(5); // wait until sensor triggers
    set_motor(0, 0);
    if (done) return false;

    // Define the near-limit position as encoder count zero.
    // All subsequent carriage positions are measured from this reference.
    enc_carriage.count.store(0);
    std::cout << "[homing] near limit found — carriage count zeroed\n";
    pause_ms(300);

    // ── Step 3: find far limit and measure total rail length ──────────────────
    std::cout << "[homing] seeking far limit...\n";
    set_motor(0, HOMING_DUTY);                             // drive right toward far stop
    while (gpioRead(PROX_FAR) == 0 && !done) pause_ms(5); // wait until far sensor triggers
    set_motor(0, 0);
    if (done) return false;

    const long long total_counts = enc_carriage.count.load();
    if (total_counts == 0) {
        // If the encoder didn't move at all, something is wrong with the hardware.
        // We can't home without a valid rail length, so abort with an error.
        std::cerr << "[homing] ERROR: encoder reported zero range — check wiring\n";
        return false;
    }
    std::cout << "[homing] far limit found — range = " << total_counts
              << " counts (" << std::fixed << std::setprecision(4)
              << total_counts * METERS_PER_COUNT << " m)\n";
    pause_ms(300);

    // ── Step 4: drive to centre ───────────────────────────────────────────────
    // The centre is halfway between the two limits. We drive left (decreasing
    // count) until we reach the midpoint, then redefine that as count zero.
    // This makes pkt.x == 0 correspond to the physical centre of the rail.
    const long long center = total_counts / 2;
    std::cout << "[homing] centering (target = " << center << " counts)...\n";
    set_motor(HOMING_DUTY, 0);   // drive left from far limit toward centre
    if (total_counts > 0) {
        while (enc_carriage.count.load() > center && !done) pause_ms(5);  // count decreases as we go left
    } else {
        while (enc_carriage.count.load() < center && !done) pause_ms(5);  // negative range: count increases left
    }
    set_motor(0, 0);
    if (done) return false;

    enc_carriage.count.store(0);   // redefine current position as x = 0
    std::cout << "[homing] carriage centered and zeroed\n";

    // ── Step 5: wait for pendulum to hang still ───────────────────────────────
    // We sample the pendulum encoder once per second into a sliding window.
    // When max - min across the last STABLE_WINDOW samples drops below STABLE_TICKS,
    // the pendulum is considered stationary and we can safely zero it.
    //
    // Why a sliding window instead of just waiting a fixed time?
    // The pendulum might take a variable amount of time to damp depending on
    // its initial swing amplitude. A fixed wait could be too short (pendulum
    // still moving) or wastefully long (already still after 3 seconds).
    std::cout << "[homing] waiting for pendulum to settle — press ENTER to abort\n";
    std::deque<long long> window;   // rolling buffer of pendulum encoder readings, one per second
    for (int elapsed = 0; !done; ++elapsed) {
        std::cout << "\r[homing] settling: " << std::setw(4) << elapsed
                  << "s elapsed...   " << std::flush;
        pause_ms(1000);   // sample once per second

        window.push_back(enc_pendulum.count.load());   // add latest reading to the back
        if ((int)window.size() > STABLE_WINDOW)
            window.pop_front();   // evict the oldest reading to keep the window fixed length

        if ((int)window.size() == STABLE_WINDOW) {   // only check stability once window is full
            long long lo = *std::min_element(window.begin(), window.end());
            long long hi = *std::max_element(window.begin(), window.end());
            if (hi - lo < STABLE_TICKS) break;   // variation is within threshold — pendulum is still
        }
    }
    if (done) return false;
    std::cout << "\n[homing] pendulum settled\n";

    // ── Step 6: zero pendulum encoder ────────────────────────────────────────
    // Define the current hanging position as theta = 0. The RL client's reward
    // function and the policy both use theta = 0 for "hanging down" as the reference.
    enc_pendulum.count.store(0);
    std::cout << "[homing] pendulum encoder zeroed — homing complete\n\n";
    return true;
}

// ── Fast centre-only re-home ──────────────────────────────────────────────────
// Used between most RL episodes to save time. Skips the rail-limit scan and
// drives directly back to encoder zero (the centre from the last full homing).
bool homing_center_only(EncoderState& enc_carriage, EncoderState& enc_pendulum,
                        std::atomic<bool>& done)
{
    std::cout << "[homing] fast re-home: driving to center\n";

    // ── Back off if touching either limit ─────────────────────────────────────
    // If the episode ended at a limit, we need to back off before driving to centre.
    if (gpioRead(PROX_NEAR) != 0) {
        std::cout << "[homing] at near limit — backing off\n";
        set_motor(0, HOMING_DUTY);
        while (gpioRead(PROX_NEAR) != 0 && !done) pause_ms(5);
        set_motor(0, 0);
        pause_ms(300);
    } else if (gpioRead(PROX_FAR) != 0) {
        std::cout << "[homing] at far limit — backing off\n";
        set_motor(HOMING_DUTY, 0);
        while (gpioRead(PROX_FAR) != 0 && !done) pause_ms(5);
        set_motor(0, 0);
        pause_ms(300);
    }
    if (done) return false;

    // ── Drive to encoder zero (centre) ───────────────────────────────────────
    // After a full homing, encoder zero is the physical rail centre.
    // We drive toward it based on the sign of the current count:
    //   positive count = right of centre → drive left
    //   negative count = left of centre  → drive right
    long long pos = enc_carriage.count.load();
    if (pos > 0) {
        set_motor(HOMING_DUTY, 0);   // drive left
        while (enc_carriage.count.load() > 0 && !done) pause_ms(5);
    } else if (pos < 0) {
        set_motor(0, HOMING_DUTY);   // drive right
        while (enc_carriage.count.load() < 0 && !done) pause_ms(5);
    }
    set_motor(0, 0);
    if (done) return false;

    enc_carriage.count.store(0);   // snap the position to exactly zero to correct for overshoot
    std::cout << "[homing] carriage centered\n";

    // ── Wait for pendulum to settle ───────────────────────────────────────────
    // Same stability logic as in homing(), but sampled at 5 Hz (every 200 ms)
    // instead of 1 Hz. The faster sampling rate prevents aliasing: a 1 Hz pendulum
    // swing sampled at 1 Hz could appear stationary if the samples happen to land
    // at the same phase each time. Sampling at 5 Hz gives 5 samples per swing,
    // reliably capturing the motion. 50 samples × 200 ms = 10 s minimum wait.
    constexpr int SETTLE_SAMPLES     = 50;
    constexpr int SETTLE_INTERVAL_MS = 200;

    std::cout << "[homing] waiting for pendulum to settle — press ENTER to abort\n";
    std::deque<long long> window;
    int elapsed_ms = 0;
    while (!done) {
        pause_ms(SETTLE_INTERVAL_MS);
        elapsed_ms += SETTLE_INTERVAL_MS;
        std::cout << "\r[homing] settling: " << std::setw(4) << (elapsed_ms / 1000)
                  << "s elapsed...   " << std::flush;

        window.push_back(enc_pendulum.count.load());
        if ((int)window.size() > SETTLE_SAMPLES)
            window.pop_front();
        if ((int)window.size() == SETTLE_SAMPLES) {
            long long lo = *std::min_element(window.begin(), window.end());
            long long hi = *std::max_element(window.begin(), window.end());
            if (hi - lo < STABLE_TICKS) break;
        }
    }
    if (done) return false;
    std::cout << "\n[homing] pendulum settled\n";

    enc_pendulum.count.store(0);   // re-zero the pendulum at its current hanging position
    std::cout << "[homing] pendulum encoder zeroed — re-home complete\n\n";
    return true;
}
