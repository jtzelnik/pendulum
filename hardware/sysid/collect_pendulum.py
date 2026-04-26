"""
Collect free-oscillation data from the pendulum for parameter identification.

The cart is coasted (duty=0) throughout; hold it still by hand or accept slight
drift. Give the pendulum a small push (~0.1–0.2 rad) to start it oscillating
near the hang position (θ=0). The script records θ(t) and θ_dot(t) for a fixed
duration, then saves a CSV used by fit_params.py to estimate ω and kv.

Usage:
    cd hardware/sysid
    python collect_pendulum.py
    python collect_pendulum.py --duration 15 --output my_osc.csv
"""

import sys
import csv
import math
import select
import argparse
import yaml
from pathlib import Path

# Reuse transport and protocol from the RL folder — no copy needed.
sys.path.insert(0, str(Path(__file__).parent.parent / "rl"))
from zmq_client import ZMQClient   # noqa: E402

_THETA_DOT_LIMIT = 10.0   # stop early if |θ_dot| exceeds this (rad/s) to avoid LLI auto-home
_X_WARN          = 0.25   # print a warning if |x| exceeds this (m) — cart has drifted


def load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "rl" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record pendulum free-oscillation data for sysid."
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="recording duration in seconds (default: 10)")
    parser.add_argument("--output", default="pendulum_osc.csv",
                        help="output CSV filename (default: pendulum_osc.csv)")
    args = parser.parse_args()

    cfg     = load_cfg()
    conn    = cfg["connection"]
    loop_hz = cfg["loop_hz"]

    client = ZMQClient(conn["host"], conn["port_state"], conn["port_cmd"])

    print(f"Connecting to Pi at {conn['host']} ...")
    if not client.poll(30_000):
        print("ERROR: LLI not responding after 30 s — is the Pi running?")
        client.close()
        return

    client.flush()
    print("Connected.\n")

    # Live readout — let the user see the current angle while positioning the pendulum.
    # Press Enter to end the live view and immediately begin recording.
    print("Live readings — hold the cart still and position the pendulum at your desired")
    print("starting angle (~0.1–0.2 rad from vertical). Press Enter when ready to release.\n")
    while True:
        client.send_cmd(0)
        pkt = client.recv_state()
        theta_deg = math.degrees(pkt.theta)
        print(
            f"\r  θ={pkt.theta:+.4f} rad ({theta_deg:+5.1f}°)  "
            f"θ_dot={pkt.theta_dot:+6.2f} rad/s  "
            f"x={pkt.x:+.4f} m  x_dot={pkt.x_dot:+.3f} m/s",
            end="", flush=True,
        )
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()   # consume the newline
            break

    print(f"\n\nRecording for {args.duration} s  (stop early if |θ_dot| > {_THETA_DOT_LIMIT} rad/s) ...")

    rows      = []
    t_origin  = None
    n_ticks   = int(args.duration * loop_hz) + 5   # a few extra ticks as buffer
    early_stop = False

    try:
        for _ in range(n_ticks):
            client.send_cmd(0)               # coast — no force on cart
            pkt = client.recv_state()        # block one tick (~1/loop_hz s)

            t_us = pkt.timestamp_us
            if t_origin is None:
                t_origin = t_us

            t_s = (t_us - t_origin) / 1e6

            if t_s > args.duration:
                break

            rows.append((t_s, pkt.theta, pkt.theta_dot))

            if abs(pkt.x) > _X_WARN:
                print(f"\r  WARNING: cart at x={pkt.x:+.3f} m — hold it closer to centre", flush=True)

            if abs(pkt.theta_dot) > _THETA_DOT_LIMIT:
                print(f"\r  Early stop: |θ_dot| = {abs(pkt.theta_dot):.1f} rad/s > {_THETA_DOT_LIMIT}", flush=True)
                early_stop = True
                break

            print(f"\r  t={t_s:5.2f}s  θ={pkt.theta:+.4f} rad  θ_dot={pkt.theta_dot:+6.2f} rad/s  x={pkt.x:+.3f} m",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        client.send_cmd(0, estop=True)
        client.close()

    print(f"\r{' ' * 80}\r", end="")

    if not rows:
        print("No data collected.")
        return

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "theta_rad", "theta_dot_rads"])
        w.writerows(rows)

    status = "early stop" if early_stop else "complete"
    print(f"Saved {len(rows)} rows → {out_path}  ({status})")


if __name__ == "__main__":
    main()
