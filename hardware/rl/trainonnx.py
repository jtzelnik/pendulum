"""
Training entry point that starts from an ONNX checkpoint.

Differences from train.py:
  - Takes a positional ONNX path argument instead of --checkpoint.
  - At startup, converts the ONNX to PyTorch weights and immediately verifies
    the converted network produces numerically identical Q-values to the raw
    ONNX session on a set of canonical states.  If they diverge the script
    aborts before touching the hardware.
  - All checkpoints and the CSV log are written into the same folder as the
    ONNX file (not the default checkpoints/ dir).

Usage:
    cd hardware/rl
    python trainonnx.py ../onnx/cart_net.onnx
"""

import argparse
import csv
import datetime
import math
import random
import sys
import yaml
import numpy as np
import torch
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnxruntime   (required by trainonnx.py)")

from zmq_client import ZMQClient
from env import PendulumEnv
from dqn import DQNAgent


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Canonical states used to verify the conversion at startup.
# Each is (label, [sin_theta, cos_theta, theta_dot, x, x_dot]).
_VERIFY_STATES = [
    ("upright centred still",  [math.sin(0),      math.cos(0),      0.0,  0.0,  0.0]),
    ("tilt R +0.2 rad",        [math.sin(0.2),    math.cos(0.2),    0.0,  0.0,  0.0]),
    ("tilt L -0.2 rad",        [math.sin(-0.2),   math.cos(-0.2),   0.0,  0.0,  0.0]),
    ("upright thdot=+1",       [math.sin(0),      math.cos(0),      1.0,  0.0,  0.0]),
    ("upright thdot=-1",       [math.sin(0),      math.cos(0),     -1.0,  0.0,  0.0]),
    ("upright x=+0.3m",        [math.sin(0),      math.cos(0),      0.0,  0.3,  0.0]),
    ("upright x=-0.3m",        [math.sin(0),      math.cos(0),      0.0, -0.3,  0.0]),
    ("hanging down still",     [math.sin(math.pi),math.cos(math.pi),0.0,  0.0,  0.0]),
]

ACTIONS = ["L", "·", "R"]


def verify_conversion(onnx_path: str, agent: DQNAgent) -> bool:
    """Compare ONNX and PyTorch Q-values on canonical states.

    Returns True if all states match within tolerance, False otherwise.
    Prints a table so mismatches are visible at a glance.
    """
    sess = ort.InferenceSession(onnx_path)
    tol  = 1e-4
    all_ok = True

    print(f"\n{'State':<26}  ONNX act  PT act  Max Δ     OK?")
    print("-" * 64)
    for label, s in _VERIFY_STATES:
        sa      = np.array([s], dtype=np.float32)
        q_onnx  = sess.run(None, {"input_1": sa})[0][0]
        with torch.no_grad():
            q_pt = agent.policy_net(
                torch.tensor(sa).to(agent.device)
            ).cpu().numpy()[0]

        max_diff  = float(np.abs(q_onnx - q_pt).max())
        ok        = max_diff < tol
        all_ok    = all_ok and ok
        act_onnx  = ACTIONS[int(np.argmax(q_onnx))]
        act_pt    = ACTIONS[int(np.argmax(q_pt))]
        flag      = "OK" if ok else "MISMATCH"
        print(f"{label:<26}  {act_onnx:<9} {act_pt:<7} {max_diff:.2e}  {flag}")

    print("-" * 64)
    if all_ok:
        print("Conversion verified — PyTorch network matches ONNX exactly.\n")
    else:
        print("CONVERSION MISMATCH — aborting.  "
              "Check that hidden_sizes in config.yaml matches the ONNX architecture.\n")
    return all_ok


def run_inference(env: PendulumEnv, agent: DQNAgent, max_steps: int):
    """One greedy evaluation episode (no exploration)."""
    obs   = env.reset()
    total = 0.0
    steps = 0
    done  = False
    for _ in range(max_steps):
        action = agent.select_action(obs, greedy=True)
        obs, reward, done, _ = env.step(action)
        total += reward
        steps += 1
        theta_deg  = math.degrees(math.atan2(obs[0], obs[1]))
        action_chr = ACTIONS[action]
        print(
            f"\r  [inf] step {steps:4d}/{max_steps}"
            f"  [{action_chr}]  θ={theta_deg:+6.1f}°  dθ={obs[2]:+6.2f}r/s"
            f"  x={obs[3]:+.3f}m  dx={obs[4]:+5.2f}m/s"
            f"  r={reward:+.4f}  Σ={total:+8.2f}",
            end="", flush=True,
        )
        if done:
            break
    print(f"\r{' '*100}\r", end="")
    return total / max_steps, done


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume DQN training from an ONNX checkpoint."
    )
    parser.add_argument(
        "onnx",
        help="path to the .onnx file to load (e.g. ../onnx/cart_net.onnx)",
    )
    args = parser.parse_args()

    onnx_path = str(Path(args.onnx).resolve())
    ckpt_dir  = Path(onnx_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(__file__).parent / "config.yaml"
    cfg      = load_cfg(cfg_path)

    loop_hz = cfg["loop_hz"]
    conn    = cfg["connection"]
    hw      = cfg["hardware"]
    ep      = cfg["episode"]
    dqn_cfg = cfg["dqn"]
    net_cfg = cfg["network"]
    tr      = cfg["training"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Convert ONNX → .pt alongside the ONNX file.
    pt_path = str(ckpt_dir / (Path(onnx_path).stem + "_from_onnx.pt"))
    DQNAgent.load_onnx(onnx_path, pt_path, net_cfg["hidden_sizes"])

    agent = DQNAgent(
        hidden_sizes           = net_cfg["hidden_sizes"],
        lr                     = dqn_cfg["learning_rate"],
        epsilon                = dqn_cfg["epsilon"],
        gamma                  = dqn_cfg["gamma"],
        buffer_size            = dqn_cfg["buffer_size"],
        batch_size             = dqn_cfg["batch_size"],
        target_update_interval = dqn_cfg["target_update_interval"],
        device                 = device,
    )
    agent.load(pt_path)

    # Abort if conversion is not numerically faithful.
    if not verify_conversion(onnx_path, agent):
        sys.exit(1)

    log_path   = ckpt_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "total_steps", "ep_len", "norm_ret", "mean_loss", "episode_status"])
    print(f"Logging to {log_path}")

    # Start counters past warmup — the ONNX weights are already trained.
    total_steps = tr["warmup_steps"]
    episode     = 0
    best_return = -float("inf")

    next_eval_ep = tr["eval_interval"]

    print(f"Connecting to Pi at {conn['host']}:{conn['port_state']}/{conn['port_cmd']}  —  press Ctrl+C to abort\n")

    client = ZMQClient(conn["host"], conn["port_state"], conn["port_cmd"])
    env = PendulumEnv(
        client        = client,
        duty          = hw["duty"],
        x_max         = hw["x_max"],
        max_steps     = ep["max_steps"],
        limit_penalty = ep["limit_penalty"],
        loop_hz       = loop_hz,
    )

    print(f"Training for {tr['max_steps']} env steps.  "
          f"Warmup skipped (ONNX weights pre-trained).  "
          f"Inference every {tr['eval_interval']} episodes.")

    try:
        while total_steps < tr["max_steps"]:
            obs       = env.reset()
            ep_steps  = 0
            ep_return = 0.0

            while True:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.buffer.add(obs, action, reward, next_obs, done)
                obs         = next_obs
                ep_steps   += 1
                total_steps += 1
                ep_return  += reward

                theta_deg  = math.degrees(math.atan2(obs[0], obs[1]))
                action_chr = ACTIONS[action]
                print(
                    f"\r  ep {episode+1:4d}  step {ep_steps:3d}/{ep['max_steps']}"
                    f"  [{action_chr}]  θ={theta_deg:+6.1f}°  dθ={obs[2]:+6.2f}r/s"
                    f"  x={obs[3]:+.3f}m  dx={obs[4]:+5.2f}m/s"
                    f"  r={reward:+.4f}  Σ={ep_return:+8.2f}",
                    end="", flush=True,
                )

                if total_steps % dqn_cfg["target_update_interval"] == 0:
                    agent.update_target()

                if done:
                    break

            episode += 1
            norm_ret = ep_return / ep["max_steps"]
            print(f"\r{' '*100}\r", end="")
            print(f"ep {episode:4d} | steps {total_steps:6d} | "
                  f"len {ep_steps:3d} | norm_ret {norm_ret:+.3f} | "
                  f"status {info['episode_status']}")

            mean_loss = 0.0
            if len(agent.buffer) >= dqn_cfg["batch_size"]:
                losses    = [agent.train_step() for _ in range(ep_steps)]
                mean_loss = float(np.mean(losses))
                print(f"         loss {mean_loss:.4f}")

            log_writer.writerow([episode, total_steps, ep_steps,
                                  f"{norm_ret:.6f}", f"{mean_loss:.6f}",
                                  info["episode_status"]])
            log_file.flush()

            if episode >= next_eval_ep:
                inf_ret, _ = run_inference(env, agent, ep["max_steps"])
                print(f"\n  [inference @ {total_steps}] norm_ret {inf_ret:+.3f}")

                ckpt = ckpt_dir / f"dqn_{total_steps:06d}.pt"
                agent.save(str(ckpt), total_steps=total_steps,
                           episode=episode, best_return=best_return)

                if inf_ret > best_return:
                    best_return = inf_ret
                    agent.save(str(ckpt_dir / "best.pt"),
                               total_steps=total_steps, episode=episode,
                               best_return=best_return)
                    print(f"  new best: {best_return:.3f}  →  best.pt\n")

                next_eval_ep += tr["eval_interval"]

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        env.estop()
        client.close()
        agent.save(str(ckpt_dir / "final.pt"),
                   total_steps=total_steps, episode=episode, best_return=best_return)
        print("Saved final.pt")
        log_file.close()


if __name__ == "__main__":
    main()
