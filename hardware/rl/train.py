"""
Training entry point — run this on the PC while the LLI is running on the Pi.

How the training loop works (high level):

  The agent alternates between two activities: collecting experience on the
  real hardware, and updating the neural network from that experience.

  Outer loop: runs episodes until total_steps reaches max_steps (150 000).
    1. env.reset()  — blocks until the Pi finishes homing the carriage
                      (~10–120 s depending on where the carriage is).
    2. Episode collection (inner loop, one iteration per 20 Hz tick):
         - Select an action with the ε-greedy policy (random with prob ε,
           otherwise ask the network).
         - Send the action to the Pi; block ~50 ms for the response.
         - Store the (state, action, reward, next_state, done) transition
           in the replay buffer for later training.
         - Every target_update_interval (1000) env steps, copy policy-net
           weights into the frozen target net (fixed Q-targets update).
    3. End-of-episode training: run one gradient step per episode step.
         - E.g. a 600-step episode → 600 calls to agent.train_step().
         - Each call draws a fresh random 1024-sample batch from the buffer.
         - This ensures the network gets more updates from longer, more
           informative episodes.
    4. Inference evaluation: every eval_interval (5000) env steps, run
       one fully greedy episode (no random actions) and report the
       normalised return.  This is the honest measure of policy quality.
       A checkpoint is saved after every evaluation.

  Warmup: for the first warmup_steps environment steps the agent ignores
    the network and picks random actions.  This pre-populates the replay
    buffer with diverse transitions before training starts, so the first
    gradient steps have meaningful data to learn from.

Usage:
    cd hardware/rl
    python train.py
"""

import argparse                 # optional --checkpoint argument for resuming training
import csv                      # CSV writer for per-episode training log
import datetime                # timestamp for log filename
import math                    # atan2 / degrees for live angle display
import random                  # randrange for warmup random actions
import yaml                    # parse config.yaml — PyYAML
import numpy as np             # np.mean for logging the per-episode loss
import torch                   # cuda availability check
from pathlib import Path       # cross-platform path construction

from zmq_client import ZMQClient   # ZeroMQ transport to the Pi
from env import PendulumEnv        # step() / reset() hardware wrapper
from dqn import DQNAgent           # network, buffer, and Bellman update


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: Path) -> dict:
    """Read and parse config.yaml into a nested dict.

    Args:
        path: Absolute or relative path to config.yaml.
    Returns:
        Nested dict matching the YAML structure.
    """
    with open(path) as f:          # open in text mode; yaml.safe_load handles encoding
        return yaml.safe_load(f)   # safe_load prevents arbitrary Python object construction from the YAML


def run_inference(env: PendulumEnv, agent: DQNAgent, max_steps: int):
    """Run one evaluation episode with the pure greedy policy — no exploration.

    'Greedy' means the agent always picks the action with the highest Q-value;
    it never takes a random action.  This gives an honest measure of how well
    the trained policy performs (random actions during evaluation would make
    the result noisy and incomparable across checkpoints).

    The normalised return (norm_ret) divides the raw cumulative reward by
    max_steps.  This puts performance on a consistent scale regardless of
    episode length: a perfect episode that lasts max_steps scores near +1,
    while an episode that ends early at a limit scores much lower.

    Returns (norm_ret, done):
        norm_ret — cumulative reward / max_steps; closer to +1 is better.
        done     — True if the LLI auto-terminated (limit or ang-vel exceeded);
                   the LLI will re-home on its own.
                   False if the episode ran to max_steps; the next env.reset()
                   must request homing explicitly.
    """
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
        action_chr = ("L", "·", "R")[action]
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
    """Build all objects from config.yaml and run the full training loop."""
    parser = argparse.ArgumentParser(
        description="Train the DQN agent on the real pendulum hardware."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional path to a .pt checkpoint to resume training from "
             "(e.g. checkpoints/best.pt).  Warmup is skipped and total_steps "
             "continues from where the checkpoint left off.",
    )
    args = parser.parse_args()

    cfg_path = Path(__file__).parent / "config.yaml"   # config.yaml sits next to this script
    cfg      = load_cfg(cfg_path)                       # parse into nested dict

    loop_hz = cfg["loop_hz"]       # control loop frequency; must match the compiled LLI binary
    conn    = cfg["connection"]   # host / port_state / port_cmd
    hw      = cfg["hardware"]     # duty / x_max
    ep      = cfg["episode"]      # max_steps / limit_penalty
    dqn_cfg = cfg["dqn"]          # lr / epsilon / gamma / buffer_size / batch_size / target_update_interval
    net_cfg = cfg["network"]      # hidden_sizes
    tr      = cfg["training"]     # max_steps / eval_interval / checkpoint_dir

    print(f"Connecting to Pi at {conn['host']}:{conn['port_state']}/{conn['port_cmd']}  —  press Ctrl+C to abort\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    client = ZMQClient(                    # open ZMQ sockets to the Pi before constructing the env
        conn["host"],                      # Pi's static ethernet IP (192.168.10.2)
        conn["port_state"],                # SUB connects to LLI's PUB (5555)
        conn["port_cmd"],                  # PUSH connects to LLI's PULL (5556)
    )
    env = PendulumEnv(                     # wrap the ZMQ connection in the Gymnasium-style interface
        client        = client,            # transport layer injected here
        duty          = hw["duty"],        # PWM magnitude for left/right actions (230)
        x_max         = hw["x_max"],       # track half-length for reward and boundary (0.35 m)
        max_steps     = ep["max_steps"],   # maximum steps per episode (800)
        limit_penalty = ep["limit_penalty"],   # reward penalty on limit-sensor terminal (−400)
        loop_hz       = loop_hz,           # control loop frequency; scales poll/retry timeouts
    )
    agent = DQNAgent(                                              # construct policy net, target net, buffer, and optimiser
        hidden_sizes           = net_cfg["hidden_sizes"],          # [256, 256] from appendix S3
        lr                     = dqn_cfg["learning_rate"],         # 0.0003 — Adam learning rate
        epsilon                = dqn_cfg["epsilon"],               # 0.178 — fixed exploration rate
        gamma                  = dqn_cfg["gamma"],                 # 0.995 — discount factor
        buffer_size            = dqn_cfg["buffer_size"],           # 50 000 transitions
        batch_size             = dqn_cfg["batch_size"],            # 1 024 per gradient step
        target_update_interval = dqn_cfg["target_update_interval"],   # 1 000 env steps
        device                 = device,                           # 'cpu' or 'cuda'
    )

    ckpt_dir = Path(__file__).parent / tr["checkpoint_dir"]   # resolve checkpoint dir relative to this script
    ckpt_dir.mkdir(parents=True, exist_ok=True)               # create the directory tree if it doesn't already exist

    log_path   = ckpt_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "total_steps", "ep_len", "norm_ret", "mean_loss", "episode_status"])
    print(f"Logging to {log_path}")

    total_steps  = 0
    episode      = 0
    best_return  = -float("inf")

    if args.checkpoint:
        extras      = agent.load(args.checkpoint)
        total_steps = extras.get("total_steps", tr["warmup_steps"])   # resume step counter; default skips warmup
        episode     = extras.get("episode", 0)                         # resume episode counter
        best_return = extras.get("best_return", -float("inf"))
        print(f"Resumed from {args.checkpoint}  "
              f"(total_steps={total_steps}, episode={episode})")

    next_eval_ep = episode + tr["eval_interval"]   # first eval N episodes from now

    print(f"Training for {tr['max_steps']} env steps. "
          f"Warmup {tr['warmup_steps']} steps (random). "
          f"Inference every {tr['eval_interval']} episodes.")

    try:
        while total_steps < tr["max_steps"]:   # outer loop: run episodes until step budget is exhausted
            obs       = env.reset()   # block until LLI finishes homing; returns first observation of the episode
            ep_steps  = 0             # steps taken in this episode; used as gradient-step count at end of episode
            ep_return = 0.0           # raw cumulative reward for logging the normalised return

            # ── episode collection ────────────────────────────────────────
            while True:                                                   # inner loop: one iteration = one 20 Hz LLI tick (~50 ms)
                if total_steps < tr["warmup_steps"]:
                    # Warmup: ignore the network and pick random actions.
                    # The network's weights are random at the start, so its
                    # Q-value predictions are meaningless.  Filling the buffer
                    # with diverse random transitions first gives the first
                    # gradient steps meaningful, varied data to learn from.
                    action = random.randrange(PendulumEnv.N_ACTIONS)
                else:
                    action = agent.select_action(obs)                     # ε-greedy: random with prob ε, best Q-value otherwise
                next_obs, reward, done, info = env.step(action)           # send command, block ~50 ms, receive result
                agent.buffer.add(obs, action, reward, next_obs, done)     # store transition for later sampling
                obs        = next_obs   # advance observation for the next action selection
                ep_steps  += 1          # count steps for the gradient-step budget
                total_steps += 1        # advance the global step counter

                ep_return += reward

                theta_deg  = math.degrees(math.atan2(obs[0], obs[1]))
                action_chr = ("L", "·", "R")[action]
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
            # Divide cumulative reward by max_steps to normalise: a full episode
            # at maximum reward scores +1; a short or low-reward episode scores
            # much less, regardless of how many steps it lasted.
            norm_ret = ep_return / ep["max_steps"]
            print(f"\r{' '*100}\r", end="")   # clear the live step line
            print(f"ep {episode:4d} | steps {total_steps:6d} | "
                  f"len {ep_steps:3d} | norm_ret {norm_ret:+.3f} | "
                  f"status {info['episode_status']}")

            # ── end-of-episode training ───────────────────────────────────
            # Run one gradient step per episode step.  A 600-step episode
            # triggers 600 train_step() calls; each draws an independent
            # random batch from the replay buffer.  Tying the gradient budget
            # to episode length means more experience → more learning updates,
            # while very short episodes (early bad episodes) don't over-train.
            mean_loss = 0.0
            if len(agent.buffer) >= dqn_cfg["batch_size"]:
                losses = [agent.train_step() for _ in range(ep_steps)]
                mean_loss = float(np.mean(losses))
                print(f"         loss {mean_loss:.4f}")

            log_writer.writerow([episode, total_steps, ep_steps, f"{norm_ret:.6f}", f"{mean_loss:.6f}", info["episode_status"]])
            log_file.flush()

            # ── inference evaluation ──────────────────────────────────────
            if episode >= next_eval_ep:
                inf_ret, _ = run_inference(env, agent, ep["max_steps"])
                print(f"\n  [inference @ {total_steps}] norm_ret {inf_ret:+.3f}")

                ckpt = ckpt_dir / f"dqn_{total_steps:06d}.pt"
                agent.save(str(ckpt), total_steps=total_steps, episode=episode, best_return=best_return)

                if inf_ret > best_return:
                    best_return = inf_ret
                    agent.save(str(ckpt_dir / "best.pt"),
                               total_steps=total_steps, episode=episode, best_return=best_return)
                    print(f"  new best: {best_return:.3f}  →  best.pt\n")

                next_eval_ep += tr["eval_interval"]

    except KeyboardInterrupt:   # Ctrl+C pressed — exit gracefully rather than leaving the motor running
        print("\nInterrupted.")

    finally:                                            # always executed, even on exception
        env.estop()                                     # send emergency-stop so the motor halts immediately
        client.close()                                  # tear down ZMQ sockets cleanly
        agent.save(str(ckpt_dir / "final.pt"),
                   total_steps=total_steps, episode=episode, best_return=best_return)
        print("Saved final.pt")
        log_file.close()                                # flush and close the CSV log


if __name__ == "__main__":
    main()   # entry point when the script is run directly: python train.py
