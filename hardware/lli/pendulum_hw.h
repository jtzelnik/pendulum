#pragma once                   // tells the compiler to include this file only once, even if #included from multiple places
#include <pigpio.h>            // pigpio library: provides GPIO control, PWM, and interrupt callbacks on the Raspberry Pi
#include <atomic>              // std::atomic: allows variables to be safely read/written from multiple threads without locks
#include <cstdint>             // fixed-width integer types: int8_t (8-bit signed), int64_t (64-bit signed), etc.

// ── Pin assignments ──────────────────────────────────────────────────────────
// These are BCM (Broadcom chip) GPIO numbers — NOT the physical pin numbers
// printed on the Pi header.
inline constexpr unsigned ENC1_A     = 17, ENC1_B    = 27;  // carriage encoder: quadrature channel A and B
inline constexpr unsigned ENC2_A     = 22, ENC2_B    = 26;  // pendulum encoder: quadrature channel A and B
inline constexpr unsigned PROX_NEAR  = 21, PROX_FAR  = 20;  // proximity (limit) sensors: near stop and far stop
inline constexpr unsigned BTN_RIGHT  = 5,  BTN_LEFT  = 6;   // manual-drive pushbuttons (active-low: reads 0 when pressed)
inline constexpr unsigned MOTOR_R    = 12, MOTOR_L   = 13;  // IBT2 motor driver PWM inputs: R_PWM and L_PWM

// PWM duty cycle used for normal RL-driven motor commands.
// Range is 0–255 (pigpio software PWM scale). 230 ≈ 90% power — aggressive
// enough for snappy movement while leaving a small margin below the driver's limit.
inline constexpr unsigned MOTOR_DUTY = 230;

// ── Encoder / kinematics constants ──────────────────────────────────────────
// The encoders (E38S6-600-24G) have 600 lines per revolution. Quadrature
// decoding watches both rising and falling edges on both A and B channels,
// producing 4 × 600 = 2400 unique positions per shaft revolution.
inline constexpr long long COUNTS_PER_REV = 2400;

// ── Loop frequency ────────────────────────────────────────────────────────────
// LOOP_HZ is injected by the Makefile via -D LOOP_HZ=<n> (default 20).
// Override at build time: make LOOP_HZ=50
// Must match loop_hz in hardware/rl/config.yaml.
// Exact integer-period frequencies work cleanly: 10, 20, 25, 50, 100 Hz.
#ifndef LOOP_HZ
#define LOOP_HZ 20
#endif
inline constexpr double DT             = 1.0  / LOOP_HZ;   // seconds per tick (nominal)
inline constexpr int    LOOP_PERIOD_MS = 1000 / LOOP_HZ;   // milliseconds per tick

inline constexpr double PI = 3.14159265358979323846;

// METERS_PER_COUNT converts one encoder count into metres of carriage travel.
// The carriage belt wraps around a pulley of diameter 51 mm (circumference = π × 0.051 m).
// One full shaft revolution moves the belt exactly that far, and spans 2400 counts.
inline constexpr double METERS_PER_COUNT = (PI * 0.051) / COUNTS_PER_REV;

// RAD_PER_COUNT converts one encoder count into radians of pendulum rotation.
// The pendulum encoder sits directly on the pivot shaft (1:1 gear ratio),
// so 2400 counts = one full rotation = 2π radians.
inline constexpr double RAD_PER_COUNT = (2.0 * PI) / COUNTS_PER_REV;

// ── Butterworth velocity-filter coefficients ─────────────────────────────────
// Velocity is estimated each tick by finite difference: (position change) / (time elapsed).
// Because the encoder has discrete counts, small movements produce noisy spikes
// (e.g., 0 counts one tick then 2 counts the next). A low-pass filter smooths
// these out while keeping the estimate responsive to real motion.
//
// Filter design: 2nd-order Butterworth, Fc = LOOP_HZ/4, Fs = LOOP_HZ.
//   - Fc is always set to Fs/4 (quarter of the sample rate) regardless of LOOP_HZ.
//     This is intentional: Fc/Fs = 0.25 keeps the filter at 50 % of Nyquist for any rate,
//     so the coefficients below are CONSTANT and do not need recomputing when LOOP_HZ changes.
//   - Effective Fc scales with LOOP_HZ (e.g. 5 Hz at 20 Hz, 12.5 Hz at 50 Hz).
//   - Designed via bilinear transform: K = tan(π × Fc / Fs) = tan(π/4) = 1.0 (exact always).
//     Fc = Fs/4 is the special case where K = 1, giving the clean closed-form coefficients
//     below and eliminating the A1 feedback term (A1 = 0 exactly).
//
// The filter is a "biquad" — a two-pole IIR section — with difference equation:
//   y[n] = B0·x[n] + B1·x[n-1] + B2·x[n-2] − A1·y[n-1] − A2·y[n-2]
// where x[n] is raw velocity this tick and y[n] is the filtered output.
//
// Coefficients (exact closed forms):
//   B0 = B2 = 1 − 1/√2  ≈ 0.292893
//   B1      = 2 − √2    ≈ 0.585786
//   A1      = 0.0        (exact — consequence of Fc = Fs/4)
//   A2      = 3 − 2√2   ≈ 0.171573
//   pole magnitude = √A2 = √2 − 1 ≈ 0.414 < 1  ✓ stable
inline constexpr double VEL_B0 =  0.292893;
inline constexpr double VEL_B1 =  0.585786;
inline constexpr double VEL_B2 =  0.292893;
inline constexpr double VEL_A1 =  0.0;
inline constexpr double VEL_A2 =  0.171573;

// ── Quadrature decode table ──────────────────────────────────────────────────
// A quadrature encoder outputs two square-wave signals (A and B) that are
// 90° out of phase with each other. As the shaft turns:
//   • Forward: states cycle 00 → 01 → 11 → 10 → 00 → ...
//   • Reverse: states cycle 00 → 10 → 11 → 01 → 00 → ...
//
// We track the combined 2-bit state (A << 1 | B) and look up the direction
// each time it changes. This gives 4× the resolution of counting one channel alone.
//
// QUAD_TABLE is indexed by (prev_state << 2) | curr_state (a 4-bit value).
// Returns +1 (forward), -1 (reverse), or 0 (no change / illegal transition).
// Illegal transitions (both bits flipping simultaneously) are pre-filtered in
// the callback and never reach this table.
inline constexpr int8_t QUAD_TABLE[16] = {
     0, -1, +1,  0,   // prev=00: no-move | B falls→reverse | A rises→forward | illegal
    +1,  0,  0, -1,   // prev=01: A rises→forward | no-move | illegal | B rises→reverse
    -1,  0,  0, +1,   // prev=10: B falls→reverse | illegal | no-move | A falls→forward
     0, +1, -1,  0    // prev=11: illegal | A falls→forward | B rises→reverse | no-move
};

// ── Encoder state ────────────────────────────────────────────────────────────
// Holds everything needed to track one quadrature encoder.
// Two threads touch this struct:
//   1. The pigpio alert thread — writes count, a, b, prev via the encoder callback.
//   2. The main/control thread — reads count to compute position and velocity.
// count is declared atomic so the main thread always sees a complete 64-bit
// value and never catches it half-updated (a real risk on 32-bit ARM).
struct EncoderState {
    std::atomic<long long> count{0};   // cumulative signed tick count; positive = right/forward
    int a{0}, b{0}, prev{0};           // most recent A level, B level, and combined 2-bit previous state
    unsigned pin_a, pin_b;             // BCM GPIO numbers for this encoder's A and B channels

    // Constructor just stores the pin numbers; count and state start at zero.
    // The main program snapshots the actual pin levels before enabling callbacks
    // so the first decoded edge starts from a known-correct state.
    EncoderState(unsigned pa, unsigned pb) : pin_a(pa), pin_b(pb) {}
};

// ── Motor drive ──────────────────────────────────────────────────────────────
// The IBT2 driver has two independent PWM inputs: R_PWM spins the motor one
// way, L_PWM spins it the other. Setting both non-zero at once causes the
// driver to fight itself (shoot-through), so only one should be non-zero.
// Duty range is 0 (stopped) to 255 (full speed) for pigpio software PWM.
inline void set_motor(unsigned duty_r, unsigned duty_l) {
    gpioPWM(MOTOR_R, duty_r);   // apply R_PWM duty to the right-drive pin
    gpioPWM(MOTOR_L, duty_l);   // apply L_PWM duty to the left-drive pin
}
