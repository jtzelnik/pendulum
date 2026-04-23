#include "pendulum_hw.h"   // pin assignments, EncoderState, QUAD_TABLE, set_motor
#include "homing.h"        // homing() — physical calibration before control begins
#include "lli_loop.h"      // lli_loop() — the 50 Hz ZeroMQ control loop
#include <iostream>        // std::cout, std::cerr for startup and error messages
#include <atomic>          // std::atomic: shared variables safe to read/write from multiple threads
#include <thread>          // std::thread: run the ENTER-key listener in the background
#include <csignal>         // signal(), SIGINT: catch Ctrl+C for graceful shutdown

// ── Shared stop flag ──────────────────────────────────────────────────────────
// Any thread that wants to stop the program sets done = true.
// Both homing() and lli_loop() check this flag at the top of every blocking
// loop so they can exit cleanly instead of running forever.
// std::atomic ensures writes from one thread are immediately visible to others.
std::atomic<bool> done{false};

// Cached proximity-sensor levels, updated by pigpio callbacks.
// Reading gpioRead() directly in a tight loop is fine but slightly slower;
// caching the latest interrupt-driven value is more efficient.
std::atomic<int> prox_near_val{-1};
std::atomic<int> prox_far_val{-1};

// ── Encoder callback ──────────────────────────────────────────────────────────
// Pigpio calls this function every time an encoder pin changes level (0→1 or 1→0).
// It runs on pigpio's internal alert thread — a separate OS thread from main().
//
// What quadrature encoding is:
//   The encoder has two output channels (A and B) that produce square waves
//   90° apart. At any moment the combined (A, B) state is one of: 00, 01, 10, 11.
//   As the shaft turns forward the states step 00→01→11→10→00→...
//   As it turns backward the sequence reverses. By watching both channels we
//   detect direction and count with 4× the resolution of a single channel.
//
// Parameters (fixed by the pigpio alert-function API):
//   gpio  — BCM pin number that changed
//   level — new level: 1 (high), 0 (low), or PI_TIMEOUT (watchdog, not a real edge)
//   tick  — pigpio microsecond timestamp (unused; we use steady_clock instead)
//   ud    — user-data pointer; set to EncoderState* when the callback was registered
void encoder_cb(int gpio, int level, uint32_t /*tick*/, void* ud) {
    if (level == PI_TIMEOUT) return;   // pigpio watchdog event — not a real edge, ignore it

    // Recover the EncoderState for this encoder from the user-data pointer.
    // We registered &enc1 or &enc2 when we called gpioSetAlertFuncEx(), so
    // casting back to EncoderState* is safe.
    EncoderState* e = static_cast<EncoderState*>(ud);

    // Update whichever channel triggered this callback.
    if      (gpio == (int)e->pin_a) e->a = level;
    else if (gpio == (int)e->pin_b) e->b = level;
    else return;   // unexpected pin — should never happen, but guard anyway

    int curr = (e->a << 1) | e->b;   // pack both channel levels into a 2-bit index: [A:B]
    if (curr == e->prev) return;       // state unchanged — spurious callback, no action needed

    // A valid quadrature step changes exactly ONE bit (only A or only B toggles).
    // If both bits change simultaneously (XOR result = 0b11 = 3) the transition
    // is electrically impossible in a real quadrature signal — it's noise.
    // Discard it rather than adding a wrong count.
    if ((e->prev ^ curr) == 3) { e->prev = curr; return; }

    // Look up the direction for this prev→curr transition in the decode table.
    // The table index packs prev (2 bits, high) and curr (2 bits, low) together.
    int delta = QUAD_TABLE[(e->prev << 2) | curr];   // +1, -1, or 0
    if (delta) e->count.fetch_add(delta);             // atomically add to the position counter
    e->prev = curr;                                    // remember current state for next edge
}

// ── Proximity sensor callbacks ────────────────────────────────────────────────
// Called by pigpio when the near or far limit sensor changes state.
// We just cache the new level so the main loop can read it without a gpioRead() call.
void prox_near_cb(int /*gpio*/, int level, uint32_t /*tick*/, void* /*ud*/) {
    if (level != PI_TIMEOUT) prox_near_val.store(level);
}
void prox_far_cb(int /*gpio*/, int level, uint32_t /*tick*/, void* /*ud*/) {
    if (level != PI_TIMEOUT) prox_far_val.store(level);
}

// ── SIGINT handler ────────────────────────────────────────────────────────────
// The OS delivers SIGINT to this function when the user presses Ctrl+C.
// We only set the done flag — any cleanup happens in main() after all loops exit.
void sigint_handler(int) { done = true; }

int main() {
    // pigpio must be initialised before any GPIO operation.
    // It needs root (sudo) to map the Raspberry Pi's hardware registers.
    if (gpioInitialise() < 0) {
        std::cerr << "pigpio init failed - run with sudo\n";
        return 1;
    }

    signal(SIGINT, sigint_handler);   // register Ctrl+C handler for clean shutdown

    // Create state objects for both encoders.
    // These are passed to the interrupt callbacks (via void* user-data) and to
    // the StateEstimator so both see the same atomically-updated count.
    EncoderState enc1(ENC1_A, ENC1_B);   // carriage (linear axis)
    EncoderState enc2(ENC2_A, ENC2_B);   // pendulum (angular axis)

    // Configure GPIO directions and pull resistors.
    // Encoders use differential line drivers, so no internal pull is needed (PUD_OFF).
    // Proximity sensors have their own pull-up on the sensor module (PUD_OFF).
    gpioSetMode(ENC1_A,    PI_INPUT);  gpioSetPullUpDown(ENC1_A,    PI_PUD_OFF);
    gpioSetMode(ENC1_B,    PI_INPUT);  gpioSetPullUpDown(ENC1_B,    PI_PUD_OFF);
    gpioSetMode(ENC2_A,    PI_INPUT);  gpioSetPullUpDown(ENC2_A,    PI_PUD_OFF);
    gpioSetMode(ENC2_B,    PI_INPUT);  gpioSetPullUpDown(ENC2_B,    PI_PUD_OFF);
    gpioSetMode(PROX_NEAR, PI_INPUT);  gpioSetPullUpDown(PROX_NEAR, PI_PUD_OFF);
    gpioSetMode(PROX_FAR,  PI_INPUT);  gpioSetPullUpDown(PROX_FAR,  PI_PUD_OFF);
    gpioSetMode(MOTOR_R,   PI_OUTPUT);
    gpioSetMode(MOTOR_L,   PI_OUTPUT);

    // Reject limit-sensor glitches shorter than 1000 µs (1 ms).
    // Mechanical vibration or wire noise can produce brief spikes on the sensor
    // lines. Filtering them out prevents false limit detections that would
    // stop the motor and trigger an unwanted homing cycle.
    gpioGlitchFilter(PROX_NEAR, 1000);
    gpioGlitchFilter(PROX_FAR,  1000);

    set_motor(0, 0);   // motor off before we start reading encoders

    // Snapshot the encoder pin levels BEFORE enabling callbacks.
    // This seeds enc.prev with the actual current state, so the very first
    // edge decoded by encoder_cb transitions from a known-correct starting point.
    // Without this, prev would be 0 (default) and the first decoded edge could
    // be wrong if the encoder happens to be in state 01, 10, or 11 at startup.
    enc1.a = gpioRead(ENC1_A); enc1.b = gpioRead(ENC1_B);
    enc1.prev = (enc1.a << 1) | enc1.b;
    enc2.a = gpioRead(ENC2_A); enc2.b = gpioRead(ENC2_B);
    enc2.prev = (enc2.a << 1) | enc2.b;

    // Seed the proximity-sensor caches before enabling their callbacks.
    prox_near_val.store(gpioRead(PROX_NEAR));
    prox_far_val.store(gpioRead(PROX_FAR));

    // Register edge callbacks. From this point on, any level change on these pins
    // fires the corresponding function on pigpio's internal alert thread.
    // We pass &enc1 or &enc2 as the user-data pointer so encoder_cb knows
    // which encoder state to update without using global variables.
    gpioSetAlertFuncEx(ENC1_A,    encoder_cb,   &enc1);
    gpioSetAlertFuncEx(ENC1_B,    encoder_cb,   &enc1);
    gpioSetAlertFuncEx(ENC2_A,    encoder_cb,   &enc2);
    gpioSetAlertFuncEx(ENC2_B,    encoder_cb,   &enc2);
    gpioSetAlertFuncEx(PROX_NEAR, prox_near_cb, nullptr);
    gpioSetAlertFuncEx(PROX_FAR,  prox_far_cb,  nullptr);

    // Spawn a background thread that waits for the user to press ENTER.
    // std::cin.get() blocks until a newline arrives; then it sets done = true.
    // This gives a way to stop the program gracefully from the terminal.
    std::thread input_thread([]() { std::cin.get(); done = true; });

    // Run the full startup homing sequence. This physically moves the carriage
    // to find the rail limits, centres it, and waits for the pendulum to hang
    // still before zeroing both encoders. Without homing, position readings
    // are relative to wherever the hardware happened to be at power-on.
    // Returns false if the user interrupts (ENTER or Ctrl+C) during homing.
    if (!homing(enc1, enc2, done)) {
        set_motor(0, 0);
        input_thread.join();
        gpioTerminate();
        return 1;
    }

    // Hand off to the 50 Hz ZeroMQ control loop. This function runs until
    // done is set (Ctrl+C or ENTER), then returns so we can clean up.
    lli_loop(enc1, enc2, done);

    set_motor(0, 0);      // ensure motor is off
    input_thread.join();  // wait for the ENTER-key thread to finish
    gpioTerminate();      // release all pigpio GPIO resources
    return 0;
}
