"""
Fit physical parameters from sysid data collected by collect_pendulum.py and
collect_cart.py.

Pendulum model (damped harmonic oscillator near θ=0):
    θ(t) = A · exp(-α·t) · cos(ω_d·t + φ)

    Fitted params: A, α, ω_d, φ
    Reported:      ω  = sqrt(ω_d² + α²)   (natural frequency, rad/s)
                   kv_proxy = 2·α          (= kv/J; true kv requires inertia J)

Cart model (first-order velocity response per step):
    x_dot(t) = x_dot_ss · (1 − exp(-t/τ))

    Fitted per trial: τ, x_dot_ss
    Then across trials: kU, fc, fd  from  x_dot_ss = (kU·|d| − fc) / (1 + fd)

Usage:
    cd hardware/sysid
    python fit_params.py
    python fit_params.py --pendulum my_osc.csv --cart my_step.csv
    python fit_params.py --no_plots
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit

_HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend; works without a display
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    pass


# ── Pendulum ──────────────────────────────────────────────────────────────────

def _pend_model(t, A, alpha, omega_d, phi):
    return A * np.exp(-alpha * t) * np.cos(omega_d * t + phi)


def fit_pendulum(csv_path: Path) -> dict:
    """Fit damped oscillator to θ(t) data; return dict of parameter estimates."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t    = data[:, 0]
    th   = data[:, 1]

    if len(t) < 10:
        raise ValueError(f"Too few rows in {csv_path} ({len(t)}) to fit.")

    # Initial guess: amplitude from data range, frequency from zero-crossings.
    A0       = (th.max() - th.min()) / 2.0
    # Count zero crossings for frequency estimate.
    zc       = np.where(np.diff(np.sign(th)))[0]
    if len(zc) >= 2:
        omega_d0 = math.pi * (len(zc) - 1) / (t[zc[-1]] - t[zc[0]] + 1e-9)
    else:
        omega_d0 = 3.0   # fallback ~0.5 Hz
    alpha0  = 0.1
    phi0    = 0.0

    p0     = [A0, alpha0, omega_d0, phi0]
    bounds = ([0, 0, 0.1, -math.pi], [10, 50, 100, math.pi])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(_pend_model, t, th, p0=p0, bounds=bounds, maxfev=10_000)

    perr    = np.sqrt(np.diag(pcov))
    A, alpha, omega_d, phi = popt

    omega    = math.sqrt(omega_d**2 + alpha**2)
    kv_proxy = 2.0 * alpha

    print("\n── Pendulum fit ─────────────────────────────────────────────")
    print(f"  A        = {A:.4f}  ± {perr[0]:.4f}  rad")
    print(f"  α        = {alpha:.4f}  ± {perr[1]:.4f}  (decay rate)")
    print(f"  ω_d      = {omega_d:.4f}  ± {perr[2]:.4f}  rad/s  (damped nat. freq.)")
    print(f"  φ        = {phi:.4f}  ± {perr[3]:.4f}  rad")
    print(f"  ω        = {omega:.4f}  rad/s  (undamped nat. freq.)")
    print(f"  kv_proxy = {kv_proxy:.4f}  (= kv/J; multiply by J for true kv)")

    return {"A": A, "alpha": alpha, "omega_d": omega_d, "phi": phi,
            "omega": omega, "kv_proxy": kv_proxy,
            "t": t, "th": th, "popt": popt}


# ── Cart ──────────────────────────────────────────────────────────────────────

def _vel_rise(t, x_dot_ss, tau):
    """First-order velocity rise from 0."""
    return x_dot_ss * (1.0 - np.exp(-t / tau))


_PWM_MAX = 255.0   # 8-bit PWM range on the Pi


def _duty_to_volts(duty: float, voltage_v: float) -> float:
    """Convert a raw duty count to the voltage actually applied to the motor.

    The IBT2 is driven by 8-bit PWM (0–255). A duty count d at supply V produces
    an average motor voltage of  V_applied = (d / 255) * V_supply.
    The RL agent caps duty at 230 (≈90% of 255), so the maximum motor voltage
    during training is 0.9 * V_supply — this is already implicit in the conversion.
    """
    return (abs(duty) / _PWM_MAX) * voltage_v


def fit_cart(csv_path: Path) -> dict:
    """Fit first-order model per trial, then regress kU/fc/fd.

    When the CSV contains a voltage_v column (populated by collect_cart.py
    --voltages), duty counts are converted to applied motor voltage before
    regression so that kU is in physical units (m/s)/V.  Without voltage data
    kU is reported in (m/s)/(duty count) instead.
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                      converters={5: lambda v: float(v) if v.strip() else float("nan")})
    # columns: t_s, x_m, x_dot_ms, duty, trial_id, voltage_v
    trials = np.unique(data[:, 4].astype(int))

    # Check whether supply voltage was recorded for all trials.
    voltages_col  = data[:, 5]
    has_voltage   = not np.all(np.isnan(voltages_col))

    taus       = []
    ss_vals    = []   # (excitation, x_dot_ss) where excitation is volts or duty count
    trial_fits = []

    for tid in trials:
        mask = data[:, 4] == tid
        rows = data[mask]
        t    = rows[:, 0]
        xd   = rows[:, 2]
        duty = int(rows[0, 3])
        v_supply = float(rows[0, 5]) if has_voltage else None

        if len(t) < 5:
            continue

        # Shift so t starts at 0.
        t = t - t[0]

        # Sign: positive duty → positive velocity expected.
        sign = 1 if duty > 0 else -1
        xd_pos = sign * xd   # make all trials positive for fitting

        ss_guess  = xd_pos.max() if xd_pos.max() > 0 else 0.1
        tau_guess = t[-1] / 3.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(_vel_rise, t, xd_pos,
                                    p0=[ss_guess, tau_guess],
                                    bounds=([0, 1e-4], [10, 10]),
                                    maxfev=5000)
            xd_ss_fit, tau_fit = popt
            taus.append(tau_fit)

            if has_voltage and v_supply is not None and not math.isnan(v_supply):
                excitation = _duty_to_volts(duty, v_supply)
                v_tag = f"  V_applied={excitation:.2f}V"
            else:
                excitation = float(abs(duty))
                v_tag = ""

            ss_vals.append((excitation, xd_ss_fit))
            trial_fits.append((tid, duty, t, xd_pos, popt))
            print(f"  trial {tid:2d}  duty={duty:+4d}{v_tag}  τ={tau_fit:.3f}s  x_dot_ss={xd_ss_fit:.3f} m/s")
        except RuntimeError:
            print(f"  trial {tid:2d}  duty={duty:+4d}  FAILED to converge — skipped")

    if not taus:
        raise ValueError("No cart trials converged.")

    tau_mean = float(np.mean(taus))
    tau_std  = float(np.std(taus))

    # Least-squares: x_dot_ss = kU * excitation - fc
    # excitation is V_applied when voltage data is present, raw |duty| otherwise.
    exc_arr = np.array([s[0] for s in ss_vals], dtype=float)
    ss_arr  = np.array([s[1] for s in ss_vals], dtype=float)

    A_mat = np.column_stack([exc_arr, np.ones_like(exc_arr)])
    result, _, _, _ = np.linalg.lstsq(A_mat, ss_arr, rcond=None)
    kU_est, neg_fc = result
    fc_est = float(-neg_fc)
    fd_est = 0.0

    kU_units = "(m/s)/V" if has_voltage else "(m/s)/(duty count)"

    print("\n── Cart fit ─────────────────────────────────────────────────")
    if not has_voltage:
        print("  NOTE: no voltage data — kU is in duty-count units, not physical V.")
        print("        Re-run collect_cart.py with --voltages <V> for physical kU.")
    print(f"  τ        = {tau_mean:.4f}  ± {tau_std:.4f}  s  (mean ± std across {len(taus)} trials)")
    print(f"  kU       = {kU_est:.6f}  {kU_units}")
    print(f"  fc       = {fc_est:.4f}  m/s  (Coulomb friction, velocity units)")
    print(f"  fd       = {fd_est:.4f}  (not fitted — requires repeated experiments at same duty)")

    return {"tau": tau_mean, "tau_std": tau_std, "kU": kU_est, "kU_units": kU_units,
            "fc": fc_est, "fd": fd_est, "has_voltage": has_voltage,
            "trial_fits": trial_fits}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pendulum(res: dict, out_path: Path) -> None:
    t, th, popt = res["t"], res["th"], res["popt"]
    t_fine = np.linspace(t[0], t[-1], 500)
    th_fit = _pend_model(t_fine, *popt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, th, ".", ms=3, label="measured θ")
    ax.plot(t_fine, th_fit, label=f"fit  ω={res['omega']:.2f} rad/s  α={res['alpha']:.3f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("θ (rad)")
    ax.set_title("Pendulum free oscillation fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_cart(res: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for tid, duty, t, xd_pos, popt in res["trial_fits"]:
        t_fine = np.linspace(0, t[-1], 200)
        ax.plot(t, xd_pos, ".", ms=2, alpha=0.6)
        ax.plot(t_fine, _vel_rise(t_fine, *popt),
                label=f"d={duty:+d} τ={popt[1]:.2f}s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|x_dot| (m/s)")
    ax.set_title("Cart step-response fits")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit sysid parameters from collected CSVs."
    )
    parser.add_argument("--pendulum", default="pendulum_osc.csv",
                        help="pendulum oscillation CSV (default: pendulum_osc.csv)")
    parser.add_argument("--cart", default="cart_step.csv",
                        help="cart step-response CSV (default: cart_step.csv)")
    parser.add_argument("--output", default="params.yaml",
                        help="output parameter file (default: params.yaml)")
    parser.add_argument("--no_plots", action="store_true",
                        help="skip matplotlib plots even if matplotlib is available")
    args = parser.parse_args()

    pend_path = Path(args.pendulum)
    cart_path = Path(args.cart)
    params    = {}

    if pend_path.exists():
        pend_res = fit_pendulum(pend_path)
        params["pendulum"] = {
            "omega_rad_s": round(float(pend_res["omega"]), 5),
            "kv_proxy":    round(float(pend_res["kv_proxy"]), 5),
        }
        if _HAVE_MPL and not args.no_plots:
            plot_pendulum(pend_res, pend_path.with_suffix(".png"))
    else:
        print(f"Skipping pendulum fit — {pend_path} not found.")

    if cart_path.exists():
        cart_res = fit_cart(cart_path)
        params["cart"] = {
            "tau_s":    round(float(cart_res["tau"]), 5),
            "kU":       round(float(cart_res["kU"]), 7),
            "kU_units": cart_res["kU_units"],
            "fc":       round(float(cart_res["fc"]), 5),
            "fd":       round(float(cart_res["fd"]), 5),
        }
        if _HAVE_MPL and not args.no_plots:
            plot_cart(cart_res, cart_path.with_suffix(".png"))
    else:
        print(f"Skipping cart fit — {cart_path} not found.")

    if not params:
        print("\nNothing fitted — supply at least one CSV.")
        sys.exit(1)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"\nParameters saved → {out_path}")
    print(yaml.dump(params, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
