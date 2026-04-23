#pragma once                   // guard against multiple inclusion in the same translation unit
#include <pigpio.h>            // pigpio GPIO, PWM, and alert API
#include <atomic>              // std::atomic for thread-safe shared variables
#include <cstdint>             // fixed-width integer types (int8_t etc.)

// ── Pin assignments ──────────────────────────────────────────────────────────
inline constexpr unsigned ENC1_A     = 17, ENC1_B    = 27;  // encoder 1 quadrature lines (carriage)
inline constexpr unsigned ENC2_A     = 22, ENC2_B    = 26;  // encoder 2 quadrature lines (pendulum)
inline constexpr unsigned PROX_NEAR  = 21, PROX_FAR  = 20;  // proximity sensors: near-stop and far-stop
inline constexpr unsigned BTN_RIGHT  = 5,  BTN_LEFT  = 6;   // manual-drive pushbuttons (active-low)
inline constexpr unsigned MOTOR_R    = 12, MOTOR_L   = 13;  // IBT2 R_PWM and L_PWM inputs
inline constexpr unsigned MOTOR_DUTY = 230;                  // normal-operation PWM duty (0-255 scale)

// ── Encoder / kinematics constants ──────────────────────────────────────────
inline constexpr long long COUNTS_PER_REV   = 2400;                              // encoder pulses per full shaft revolution (4× quadrature)
inline constexpr double    DT               = 0.02;                              // nominal loop period in seconds (50 Hz)
inline constexpr double    PI               = 3.14159265358979323846;            // pi to full double precision

// 2nd-order Butterworth low-pass filter coefficients for velocity estimation.
// Design: Fs = 50 Hz, Fc = 10 Hz, bilinear transform (K = tan(π·Fc/Fs) = 0.72654).
// Difference equation: y[n] = B0·x[n] + B1·x[n-1] + B2·x[n-2] − A1·y[n-1] − A2·y[n-2]
inline constexpr double    VEL_B0 =  0.20657;
inline constexpr double    VEL_B1 =  0.41314;
inline constexpr double    VEL_B2 =  0.20657;
inline constexpr double    VEL_A1 = -0.36953;
inline constexpr double    VEL_A2 =  0.19582;
inline constexpr double    METERS_PER_COUNT = (PI * 0.051) / COUNTS_PER_REV;   // linear distance per encoder count (pulley circumference / counts)
inline constexpr double    RAD_PER_COUNT    = (2.0 * PI)   / COUNTS_PER_REV;   // angular distance per encoder count (full circle / counts)

// ── Quadrature decode table ──────────────────────────────────────────────────
// Indexed by (prev_state << 2) | curr_state where state = (A << 1) | B.
// Returns +1, -1, or 0 for valid forward, reverse, or no-change transitions.
// Illegal two-bit flips (diagonal entries) are filtered before this lookup.
inline constexpr int8_t QUAD_TABLE[16] = {
     0, -1, +1,  0,   // prev=00: no-move, B-fall (reverse), A-rise (forward), illegal
    +1,  0,  0, -1,   // prev=01: A-rise (forward), no-move, illegal, B-rise (reverse)
    -1,  0,  0, +1,   // prev=10: B-fall (reverse), illegal, no-move, A-fall (forward)
     0, +1, -1,  0    // prev=11: illegal, A-fall (forward), B-rise (reverse), no-move
};

// ── Encoder state ────────────────────────────────────────────────────────────
// Holds the mutable state for one quadrature encoder.
// Shared between the main loop (reads count) and the pigpio alert callback (writes count).
struct EncoderState {
    std::atomic<long long> count{0};   // cumulative signed tick count, updated by interrupt callback
    int a{0}, b{0}, prev{0};           // last sampled A level, B level, and combined 2-bit state
    unsigned pin_a, pin_b;             // GPIO pin numbers for this encoder's A and B channels

    // Stores pin assignments; count and state initialised to zero/idle.
    EncoderState(unsigned pa, unsigned pb)
        : pin_a(pa), pin_b(pb) {}
};

// ── Motor drive ──────────────────────────────────────────────────────────────
// Applies independent PWM duty cycles to the two IBT2 input channels.
// duty_r drives MOTOR_R (R_PWM); duty_l drives MOTOR_L (L_PWM). Range 0-255.
// Only one channel should be non-zero at a time to avoid shoot-through.
inline void set_motor(unsigned duty_r, unsigned duty_l) {
    gpioPWM(MOTOR_R, duty_r);   // set software PWM duty on R_PWM pin
    gpioPWM(MOTOR_L, duty_l);   // set software PWM duty on L_PWM pin
}
