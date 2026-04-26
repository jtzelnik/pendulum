"""
Collect cart step-response data for parameter identification.

Applies a series of constant-duty steps while the pendulum hangs straight down
(θ≈0). Records cart position and velocity over each step, then saves a CSV
used by fit_params.py to estimate τ, fc, fd, and kU.

The cart is homed to the centre before every step. A safety margin stops the
step early if the cart approaches either rail limit.

Supply voltage levels
---------------------
If --voltages is given (e.g. --voltages 12,24), the full duty sweep is repeated
once per voltage level. The script pauses between levels and asks the user to
adjust the bench supply, then waits for confirmation before continuing. A
voltage_v column is added to the CSV so fit_params.py can separate the groups.

Usage:
    cd hardware/sysid
    python collect_cart.py
    python collect_cart.py --duties 80,120,160,200 --step_duration 40
    python collect_cart.py --voltages 12,24
"""

import sys
import csv
import time
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rl"))
from zmq_client import ZMQClient   # noqa: E402

_COAST_TICKS   = 20    # ticks to coast between steps (let cart settle)
_HOME_TIMEOUT  = 60.0  # seconds to wait for homing to complete


def load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "rl" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def wait_for_home(client: ZMQClient, loop_hz: int) -> bool:
    """Request homing and block until the first status=0 packet. Returns False on timeout."""
    tick_ms  = int(1000 / loop_hz)
    deadline = time.monotonic() + _HOME_TIMEOUT
    last_send = time.monotonic() - 1.0

    print("  [home] requesting ...", end="", flush=True)

    # Send request_home until status=3 or silence (homing in progress).
    last_packet_t = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_send >= 2.0 / loop_hz:
            client.send_cmd(0, request_home=True)
            last_send = now
        if now - last_packet_t > 4.0 / loop_hz:
            break   # silence → homing started
        if client.poll(tick_ms):
            pkt = client.recv_state()
            last_packet_t = time.monotonic()
            if pkt.episode_status in (3, 1, 2):
                break

    # Wait for homing to finish (status=0 resumes).
    print(" waiting ...", end="", flush=True)
    while time.monotonic() < deadline:
        pkt = client.recv_state()
        if pkt.episode_status == 0:
            print(f" done  x={pkt.x:+.3f} m")
            return True

    print(" TIMEOUT")
    return False


def run_step(client: ZMQClient, duty: int, n_ticks: int,
             x_safety: float, loop_hz: int, trial_id: int,
             voltage: float | None) -> list:
    """Apply a constant duty for up to n_ticks and return rows list."""
    rows      = []
    t_origin  = None
    v_label   = f"  {voltage:.0f}V" if voltage is not None else ""

    print(f"  [step]{v_label}  duty={duty:+4d}  ticks={n_ticks} ...", end="", flush=True)

    for _ in range(n_ticks):
        client.send_cmd(duty)
        pkt = client.recv_state()

        t_us = pkt.timestamp_us
        if t_origin is None:
            t_origin = t_us
        t_s = (t_us - t_origin) / 1e6

        rows.append((t_s, pkt.x, pkt.x_dot, duty, trial_id,
                     voltage if voltage is not None else ""))

        print(f"\r  [step]{v_label}  duty={duty:+4d}  t={t_s:.2f}s  x={pkt.x:+.3f}m  x_dot={pkt.x_dot:+.3f}m/s",
              end="", flush=True)

        if abs(pkt.x) >= x_safety:
            print(f"\r  [step] safety stop at x={pkt.x:+.3f} m{' ' * 20}")
            client.send_cmd(0)
            break

    else:
        print(f"\r  [step]{v_label}  duty={duty:+4d}  done  rows={len(rows)}{' ' * 20}")

    # Coast to let cart decelerate.
    for _ in range(_COAST_TICKS):
        client.send_cmd(0)
        client.recv_state()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record cart step-response data for sysid."
    )
    parser.add_argument(
        "--duties",
        default="80,120,160,200,-80,-120,-160,-200",
        help="comma-separated duty values per voltage level (default: 80,120,160,200,-80,-120,-160,-200)",
    )
    parser.add_argument(
        "--step_duration", type=int, default=60,
        help="ticks per step (default: 60 = 3 s at 20 Hz)",
    )
    parser.add_argument(
        "--voltages", default="",
        help="comma-separated supply voltage levels in V, e.g. 12,24 — "
             "script pauses between levels and asks user to adjust the bench supply",
    )
    parser.add_argument(
        "--output", default="cart_step.csv",
        help="output CSV filename (default: cart_step.csv)",
    )
    args = parser.parse_args()

    duties   = [int(d) for d in args.duties.split(",")]
    voltages = [float(v) for v in args.voltages.split(",") if v.strip()] if args.voltages else [None]
    cfg      = load_cfg()
    conn     = cfg["connection"]
    loop_hz  = cfg["loop_hz"]
    x_max    = cfg["hardware"]["x_max"]
    x_safety = x_max - 0.05   # stop 5 cm before the physical limit

    client = ZMQClient(conn["host"], conn["port_state"], conn["port_cmd"])

    print(f"Connecting to Pi at {conn['host']} ...")
    if not client.poll(30_000):
        print("ERROR: LLI not responding — is the Pi running?")
        client.close()
        return

    client.flush()
    v_label = f"  voltage levels: {voltages}" if voltages[0] is not None else ""
    print(f"Connected.  x_safety={x_safety:.2f} m  duties={duties}{v_label}\n")

    all_rows  = []
    trial_id  = 0
    aborted   = False

    try:
        for vi, voltage in enumerate(voltages):
            # Prompt user to set supply voltage before each group (skip for single run).
            if voltage is not None:
                print(f"\n{'─' * 60}")
                print(f"  Set the bench supply to {voltage:.0f} V, then press Enter to continue.")
                print(f"{'─' * 60}")
                input()

            for duty in duties:
                if not wait_for_home(client, loop_hz):
                    print("Homing timed out — aborting.")
                    aborted = True
                    break

                rows = run_step(client, duty, args.step_duration, x_safety, loop_hz, trial_id, voltage)
                all_rows.extend(rows)
                trial_id += 1

            if aborted:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        client.send_cmd(0, estop=True)
        client.close()

    if not all_rows:
        print("No data collected.")
        return

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_m", "x_dot_ms", "duty", "trial_id", "voltage_v"])
        w.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows across {trial_id} trials → {out_path}")


if __name__ == "__main__":
    main()
