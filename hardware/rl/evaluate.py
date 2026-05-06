"""
Evaluate a saved DQN checkpoint — run the trained policy on the real hardware.

No gradient updates are performed; the network weights are frozen.  This
script is used to measure how well a checkpoint performs after training is
done, or to watch the pendulum balance in real time.

Each episode starts from the standard homed initial condition (θ = 0,
x = 0, velocities = 0) as set by the LLI homing sequence.

Normalised return (norm_ret) = cumulative reward / max_steps.
    A perfect episode (pendulum perfectly upright and centred for all
    800 steps) scores near +1.  An episode that ends early at a rail
    limit scores much lower.  Dividing by max_steps makes it possible
    to compare episodes of different lengths on the same scale.

Mean ± std across episodes: the mean tells you the average policy quality;
    the standard deviation tells you how consistent it is.  A low std means
    the policy behaves reliably; a high std means it sometimes works well
    and sometimes fails.

Usage:
    cd hardware/rl
    python evaluate.py checkpoints/best.pt
    python evaluate.py checkpoints/best.pt --episodes 5
"""

import argparse          # CLI argument parsing — checkpoint path and episode count
import csv               # CSV writer for optional per-timestep logging
import datetime          # timestamped filename when --log-csv is set
import math              # atan2 / degrees for live angle display
import yaml              # parse config.yaml for all hardware / episode parameters
import torch             # device selection (CPU / CUDA)
import numpy as np       # mean / std for the summary line
from pathlib import Path  # cross-platform path resolution for config.yaml

from zmq_client import ZMQClient   # ZeroMQ transport to the Pi
from env import PendulumEnv        # step() / reset() hardware wrapper
from dqn import DQNAgent           # network weights loaded from checkpoint


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: Path) -> dict:
    """Read and parse config.yaml into a nested dict.

    Args:
        path: Path to the config.yaml file (resolved relative to this script).
    Returns:
        Nested dict matching the YAML structure.
    """
    with open(path) as f:          # open in text mode; yaml handles encoding
        return yaml.safe_load(f)   # safe_load rejects arbitrary Python object constructors


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments, load checkpoint, and run greedy evaluation episodes."""
    # argparse reads command-line arguments typed after the script name.
    # 'checkpoint' is positional (required); '--episodes' is optional with default 1.
    # Running 'python evaluate.py --help' prints a usage summary automatically.
    parser = argparse.ArgumentParser(
        description="Run a saved DQN checkpoint on the real hardware."
    )
    parser.add_argument(                                    # positional: user must supply a checkpoint path
        "checkpoint",
        help="path to .pt checkpoint file, e.g. checkpoints/best.pt",
    )
    parser.add_argument(                                    # optional: override number of eval episodes
        "--episodes",
        type=int,
        default=1,
        help="number of greedy inference episodes to run (default: 1)",
    )
    parser.add_argument(                                    # optional: write per-timestep data to CSV
        "--log-csv",
        action="store_true",
        help="write per-timestep x, x_dot, theta, theta_dot, command to a timestamped CSV file",
    )
    args = parser.parse_args()   # reads sys.argv and fills args.checkpoint, args.episodes

    cfg_path = Path(__file__).parent / "config.yaml"   # config.yaml lives alongside this script
    cfg      = load_cfg(cfg_path)                       # parse all YAML sections

    loop_hz = cfg["loop_hz"]       # control loop frequency; must match the compiled LLI binary
    conn    = cfg["connection"]   # host / port_state / port_cmd
    hw      = cfg["hardware"]     # duty / x_max
    ep      = cfg["episode"]      # max_steps / limit_penalty
    dqn_cfg = cfg["dqn"]          # lr / epsilon / gamma / batch_size / target_update_interval
    net_cfg = cfg["network"]      # hidden_sizes — must match what was used during training

    print(f"Connecting to Pi at {conn['host']}:{conn['port_state']}/{conn['port_cmd']}  —  press Ctrl+C to abort\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    client = ZMQClient(                  # open ZMQ sockets to the Pi
        conn["host"],                    # Pi IP from config.yaml
        conn["port_state"],              # SUB → Pi's PUB port (5555)
        conn["port_cmd"],                # PUSH → Pi's PULL port (5556)
    )
    env = PendulumEnv(                        # hardware environment wrapper
        client        = client,               # ZMQ transport injected
        duty          = hw["duty"],           # PWM magnitude for left/right (230)
        x_max         = hw["x_max"],          # track half-length for reward (0.35 m)
        max_steps     = ep["max_steps"],      # episode length cap (800)
        limit_penalty = ep["limit_penalty"],  # reward penalty on limit hit (−400)
        loop_hz       = loop_hz,              # control loop frequency; scales poll/retry timeouts
    )
    agent = DQNAgent(                                              # construct agent with same architecture as training
        hidden_sizes           = net_cfg["hidden_sizes"],          # must match the saved checkpoint — [256, 256]
        lr                     = dqn_cfg["learning_rate"],         # not used during eval; required by constructor
        epsilon                = dqn_cfg["epsilon"],               # not used during eval (greedy=True bypasses it)
        gamma                  = dqn_cfg["gamma"],                 # not used during eval; required by constructor
        buffer_size            = 1,                                # buffer not used during evaluation; set to 1 to minimise memory
        batch_size             = dqn_cfg["batch_size"],            # not used during eval; required by constructor
        target_update_interval = dqn_cfg["target_update_interval"],   # not used during eval; required by constructor
        device                 = device,                           # maps checkpoint tensors to the right device
    )
    ckpt_path = args.checkpoint
    if ckpt_path.endswith(".onnx"):
        out_path = str(Path(ckpt_path).with_name(Path(ckpt_path).stem + "_from_onnx.pt"))
        DQNAgent.load_onnx(ckpt_path, out_path, net_cfg["hidden_sizes"])
        ckpt_path = out_path
    agent.load(ckpt_path)                      # restore policy_net, target_net, and optimizer state
    print(f"Loaded {ckpt_path}  ({device})")   # confirm which file and device are in use

    returns    = []    # collect normalised returns across episodes for the summary
    csv_file   = None
    csv_writer = None
    if args.log_csv:
        ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path   = Path(__file__).parent / f"eval_{ts}.csv"
        csv_file   = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "step", "x", "x_dot", "theta", "theta_dot", "command", "reward"])
        print(f"Logging to {csv_path}")

    try:
        for i in range(args.episodes):          # run the requested number of evaluation episodes
            obs        = env.reset()                  # block until LLI homing completes; get first observation
            total      = 0.0                          # raw cumulative reward for this episode
            steps      = 0                            # steps taken before episode ended
            prev_theta = None                         # for incremental unwrapping across steps

            for _ in range(ep["max_steps"]):
                action = agent.select_action(obs, greedy=True)
                obs, reward, done, info = env.step(action)
                total += reward
                steps += 1
                theta_raw = math.atan2(obs[0], obs[1])
                if prev_theta is None:
                    theta_unwrapped = theta_raw
                else:
                    delta           = (theta_raw - prev_theta + math.pi) % (2 * math.pi) - math.pi
                    theta_unwrapped = prev_theta + delta
                prev_theta = theta_unwrapped
                theta_deg  = math.degrees(theta_raw)
                action_chr = ("L", "·", "R")[action]
                if csv_writer is not None:
                    csv_writer.writerow([
                        i + 1, steps,
                        obs[3], obs[4],                        # x, x_dot
                        theta_unwrapped, obs[2],               # theta (unwrapped), theta_dot
                        (-hw["duty"], 0, hw["duty"])[action],  # command
                        reward,
                    ])
                print(
                    f"\r  ep {i+1:3d}  step {steps:3d}/{ep['max_steps']}"
                    f"  [{action_chr}]  θ={theta_deg:+6.1f}°  dθ={obs[2]:+6.2f}r/s"
                    f"  x={obs[3]:+.3f}m  dx={obs[4]:+5.2f}m/s"
                    f"  r={reward:+.4f}  Σ={total:+8.2f}",
                    end="", flush=True,
                )
                if done:
                    break

            norm = total / ep["max_steps"]
            returns.append(norm)
            print(f"\r{' '*100}\r", end="")   # clear the live step line
            print(f"Episode {i+1:3d} | steps {steps:3d} | "
                  f"norm_ret {norm:+.3f} | status {info['episode_status']}")

    except KeyboardInterrupt:   # Ctrl+C — stop gracefully without leaving the motor running
        print("\nInterrupted.")

    finally:                    # always run — ensure motor is stopped even on exception
        env.estop()             # send emergency-stop command to the LLI
        client.close()          # tear down ZMQ sockets cleanly
        if csv_file is not None:
            csv_file.close()

    if returns:   # only print summary if at least one episode completed
        # mean = average normalised return across all evaluated episodes
        # std  = how much it varied — low std means the policy is consistent
        print(f"\nMean norm_ret over {len(returns)} episodes: "
              f"{np.mean(returns):+.3f}  ±  {np.std(returns):.3f}")


if __name__ == "__main__":
    main()   # entry point when run directly: python evaluate.py checkpoints/best.pt
