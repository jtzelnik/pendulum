"""
Cart sweep experiment: move carriage from one side of the track to the other
at a fixed PWM duty while recording pendulum angle response.

Useful for estimating the inertial coupling term (m·l) in the equations of
motion — the pendulum deflects in response to cart acceleration, and the
magnitude of that deflection encodes the ratio of pendulum mass times length
to total system inertia.

Procedure
---------
1. Move cart to start position (near one rail, default: left).
2. Wait for pendulum to hang still (|θ_dot| < settle_threshold).
3. Apply fixed duty across the full track and record at 20 Hz.
4. Stop when the cart approaches the far rail, then coast.
5. Save CSV.

CSV columns: t_s, x_m, x_dot_ms, theta_rad, theta_dot_rads, duty

Usage:
    cd hardware/sysid
    python collect_sweep.py --duty 120
    python collect_sweep.py --duty 80 --start right --output sweep_slow.csv
"""

import sys
import csv
import time
import math
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rl"))
from zmq_client import ZMQClient   # noqa: E402

_SETTLE_TICKS   = 20    # consecutive ticks below threshold to declare settled
_SETTLE_TIMEOUT = 30.0  # seconds before giving up waiting for settle
_STAGE_TIMEOUT  = 15.0  # seconds for move-to-side phase before aborting


def load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "rl" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def move_to_side(client: ZMQClient, duty: int, x_target: float,
                 x_safety: float, loop_hz: int) -> object:
    """Drive cart toward x_target using constant duty; stop when reached or limit.

    duty should be signed: negative to go left, positive to go right.
    Returns the last StatePacket received.
    """
    deadline = time.monotonic() + _STAGE_TIMEOUT
    pkt = None
    while time.monotonic() < deadline:
        client.send_cmd(duty)
        pkt = client.recv_state()
        theta_deg = math.degrees(pkt.theta)
        direction = "left" if duty < 0 else "right"
        print(
            f"\r  [move {direction}]  x={pkt.x:+.3f}m  target={x_target:+.3f}m"
            f"  θ={pkt.theta:+.4f}rad ({theta_deg:+.1f}°)",
            end="", flush=True,
        )
        if (duty < 0 and pkt.x <= x_target) or (duty > 0 and pkt.x >= x_target):
            break
        if abs(pkt.x) >= x_safety:
            print(f"\r  [move] safety limit reached at x={pkt.x:+.3f} m{' '*20}")
            break
    client.send_cmd(0)
    # Coast a few ticks to let the cart stop.
    for _ in range(10):
        client.send_cmd(0)
        pkt = client.recv_state()
    print(f"\r  [move done]  x={pkt.x:+.3f} m{' '*40}")
    return pkt


def wait_settle(client: ZMQClient, threshold: float, loop_hz: int) -> object:
    """Coast and wait until |θ_dot| < threshold for _SETTLE_TICKS consecutive ticks.

    Returns the last StatePacket, or None on timeout.
    """
    deadline  = time.monotonic() + _SETTLE_TIMEOUT
    count     = 0
    pkt       = None
    print(f"  [settle] waiting for |θ_dot| < {threshold:.3f} rad/s ...")
    while time.monotonic() < deadline:
        client.send_cmd(0)
        pkt = client.recv_state()
        theta_deg = math.degrees(pkt.theta)
        print(
            f"\r  [settle {count:2d}/{_SETTLE_TICKS}]"
            f"  θ_dot={pkt.theta_dot:+.4f} rad/s"
            f"  θ={pkt.theta:+.4f} rad ({theta_deg:+.1f}°)",
            end="", flush=True,
        )
        if abs(pkt.theta_dot) < threshold:
            count += 1
            if count >= _SETTLE_TICKS:
                print(f"\r  [settled]  θ={pkt.theta:+.4f} rad  θ_dot={pkt.theta_dot:+.5f} rad/s{' '*20}")
                return pkt
        else:
            count = 0
    print(f"\r  [settle] TIMEOUT after {_SETTLE_TIMEOUT:.0f} s{' '*40}")
    return None


def run_sweep(client: ZMQClient, duty: int, x_safety: float,
              loop_hz: int) -> list:
    """Apply fixed duty across the track, recording state each tick.

    Stops when x reaches the far safety boundary or episode_status != 0.
    Returns list of row tuples.
    """
    rows     = []
    t_origin = None
    sign     = 1 if duty > 0 else -1

    print(f"  [sweep] duty={duty:+d}  recording ...")

    while True:
        client.send_cmd(duty)
        pkt = client.recv_state()

        t_us = pkt.timestamp_us
        if t_origin is None:
            t_origin = t_us
        t_s = (t_us - t_origin) / 1e6

        rows.append((t_s, pkt.x, pkt.x_dot, pkt.theta, pkt.theta_dot, duty))

        theta_deg = math.degrees(pkt.theta)
        print(
            f"\r  [sweep]  t={t_s:.2f}s  x={pkt.x:+.4f}m  x_dot={pkt.x_dot:+.3f}m/s"
            f"  θ={pkt.theta:+.4f}rad ({theta_deg:+.1f}°)",
            end="", flush=True,
        )

        if pkt.episode_status != 0:
            print(f"\r  [sweep] LLI terminated (status={pkt.episode_status}){' '*30}")
            break

        # Stop before hitting the rail.
        if (duty > 0 and pkt.x >= x_safety) or (duty < 0 and pkt.x <= -x_safety):
            print(f"\r  [sweep] reached far boundary at x={pkt.x:+.3f} m{' '*30}")
            break

    client.send_cmd(0)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cart sweep: record pendulum response to full-track traversal."
    )
    parser.add_argument(
        "--duty", type=int, default=120,
        help="constant PWM duty during the sweep (default: 120)",
    )
    parser.add_argument(
        "--start", choices=["left", "right"], default="left",
        help="which side to start from (default: left)",
    )
    parser.add_argument(
        "--settle_threshold", type=float, default=0.05,
        help="|θ_dot| in rad/s below which pendulum is considered still (default: 0.05)",
    )
    parser.add_argument(
        "--output", default="cart_sweep.csv",
        help="output CSV filename (default: cart_sweep.csv)",
    )
    args = parser.parse_args()

    cfg      = load_cfg()
    conn     = cfg["connection"]
    loop_hz  = cfg["loop_hz"]
    x_max    = cfg["hardware"]["x_max"]
    x_start  = x_max - 0.06   # target start position: 6 cm from the rail
    x_safety = x_max - 0.04   # stop sweep 4 cm from rail

    # Duty signs: move-to-side uses the opposite sign from the sweep.
    if args.start == "left":
        side_duty   = -abs(args.duty)   # drive left to reach start
        sweep_duty  = +abs(args.duty)   # sweep right
        x_target    = -x_start
    else:
        side_duty   = +abs(args.duty)   # drive right to reach start
        sweep_duty  = -abs(args.duty)   # sweep left
        x_target    = +x_start

    client = ZMQClient(conn["host"], conn["port_state"], conn["port_cmd"])

    print(f"Connecting to Pi at {conn['host']} ...")
    if not client.poll(30_000):
        print("ERROR: LLI not responding — is the Pi running?")
        client.close()
        return

    client.flush()
    print(f"Connected.  duty={args.duty}  start={args.start}  "
          f"settle_threshold={args.settle_threshold} rad/s\n")

    rows = []

    try:
        # Phase 1: move to start side.
        move_to_side(client, side_duty, x_target, x_safety, loop_hz)

        # Phase 2: wait for pendulum to hang still.
        settled = wait_settle(client, args.settle_threshold, loop_hz)
        if settled is None:
            print("Pendulum did not settle — aborting. Try a smaller settle_threshold.")
            return

        print()
        input("Pendulum settled. Press Enter to start the sweep.")
        print()

        # Phase 3: sweep across track.
        rows = run_sweep(client, sweep_duty, x_safety, loop_hz)

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
        w.writerow(["t_s", "x_m", "x_dot_ms", "theta_rad", "theta_dot_rads", "duty"])
        w.writerows(rows)

    # Quick summary.
    thetas   = [r[3] for r in rows]
    max_def  = max(abs(t) for t in thetas)
    duration = rows[-1][0]
    print(f"Saved {len(rows)} rows → {out_path}")
    print(f"  duration: {duration:.2f} s   peak |θ|: {math.degrees(max_def):.2f}°")


if __name__ == "__main__":
    main()
