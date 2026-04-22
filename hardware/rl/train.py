"""
Training entry point — run this on the PC while the LLI runs on the Pi.

Loop structure (appendix Table 2 / article §DQN):

  Outer loop: episodes until total_steps reaches max_steps (150 000).
    1. env.reset()  — blocks until Pi finishes homing (~120 s first time).
    2. Episode collection: at each 50 Hz tick, select ε-greedy action,
       send to Pi, receive next state, store transition in replay buffer.
    3. Target-net sync: every target_update_interval (1000) env steps,
       copy policy-net weights into target net.
    4. End-of-episode training: run <episode_steps> gradient updates
       using the Huber loss (one update per step the episode lasted).
    5. Inference evaluation: every eval_interval (5000) env steps, run
       one greedy episode and report the normalised return.

Usage:
    cd hardware/rl
    python train.py
"""

import math                     # atan2 / degrees for live angle display
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


def run_inference(env: PendulumEnv, agent: DQNAgent, max_steps: int) -> float:
    """Execute one greedy episode and return the normalised return.

    Normalised return = cumulative_reward / max_steps, as defined in the
    article.  The maximum is 1.0 (upright, centred, full episode); the
    minimum is −0.5 (cart hits a stop on step 1, −400 penalty).

    No gradient updates are performed during inference.

    Args:
        env:       Live hardware environment.
        agent:     DQNAgent in greedy mode (ε is ignored).
        max_steps: Episode length budget used as the normalisation divisor.
    Returns:
        Normalised return in [−0.5, 1.0].
    """
    obs   = env.reset()   # wait for homing to complete and receive the first observation
    total = 0.0           # accumulate raw (un-normalised) reward over the episode
    for _ in range(max_steps):                          # cap at max_steps even if done is not raised
        action = agent.select_action(obs, greedy=True)  # greedy=True bypasses ε-random exploration
        obs, reward, done, _ = env.step(action)         # send command, receive next state and reward
        total += reward                                  # accumulate reward for normalisation
        if done:                                         # episode ended early (limit or angular-vel)
            break                                        # stop collecting; remaining steps contribute 0
    return total / max_steps   # normalised return: 1.0 = perfect episode, −0.5 = instant limit hit


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build all objects from config.yaml and run the full training loop."""
    cfg_path = Path(__file__).parent / "config.yaml"   # config.yaml sits next to this script
    cfg      = load_cfg(cfg_path)                       # parse into nested dict

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

    total_steps = 0              # cumulative environment steps across all episodes; main termination criterion
    episode     = 0              # episode counter used only for the log line
    next_eval   = tr["eval_interval"]    # next total_steps value at which to run an inference episode
    best_return = -float("inf")          # track the best inference normalised return seen so far

    print(f"Training for {tr['max_steps']} env steps. "
          f"Warmup {tr['warmup_steps']} steps (random). "
          f"Inference every {tr['eval_interval']} steps.")

    try:
        while total_steps < tr["max_steps"]:   # outer loop: run episodes until step budget is exhausted
            obs       = env.reset()   # block until LLI finishes homing; returns first observation of the episode
            ep_steps  = 0             # steps taken in this episode; used as gradient-step count at end of episode
            ep_return = 0.0           # raw cumulative reward for logging the normalised return

            # ── episode collection ────────────────────────────────────────
            while True:                                                   # inner loop: one iteration = one 50 Hz LLI tick
                if total_steps < tr["warmup_steps"]:                      # pure random exploration until buffer is seeded
                    action = random.randrange(PendulumEnv.N_ACTIONS)      # ignore network — weights are meaningless before training
                else:
                    action = agent.select_action(obs)                     # ε-greedy once buffer has enough data
                next_obs, reward, done, info = env.step(action)           # send command, block ~20 ms, receive result
                agent.buffer.add(obs, action, reward, next_obs, done)     # store transition for later sampling
                obs        = next_obs   # advance observation for the next action selection
                ep_steps  += 1          # count steps for the gradient-step budget
                total_steps += 1        # advance the global step counter

                ep_return += reward

                theta_deg  = math.degrees(math.atan2(obs[0], obs[1]))
                action_chr = ("L", "·", "R")[action]
                print(
                    f"\r  ep {episode+1:4d}  step {ep_steps:3d}/{ep['max_steps']}"
                    f"  [{action_chr}]  θ={theta_deg:+6.1f}°  x={obs[3]:+.3f}m"
                    f"  r={reward:+.4f}  Σ={ep_return:+8.2f}",
                    end="", flush=True,
                )

                if total_steps % dqn_cfg["target_update_interval"] == 0:
                    agent.update_target()

                if done:
                    break

            episode += 1
            norm_ret = ep_return / ep["max_steps"]
            print(f"\r{' '*100}\r", end="")   # clear the live step line
            print(f"ep {episode:4d} | steps {total_steps:6d} | "
                  f"len {ep_steps:3d} | norm_ret {norm_ret:+.3f} | "
                  f"status {info['episode_status']}")

            # ── end-of-episode training ───────────────────────────────────
            if len(agent.buffer) >= dqn_cfg["batch_size"]:     # only train once the buffer has enough data for a full batch
                losses = [agent.train_step() for _ in range(ep_steps)]   # one gradient step per episode step (appendix Table 2)
                print(f"         loss {np.mean(losses):.4f}")             # log mean Huber loss over this batch of gradient steps

            # ── inference evaluation ──────────────────────────────────────
            if total_steps >= next_eval:                          # triggered once per eval_interval env steps
                inf_ret = run_inference(env, agent, ep["max_steps"])      # one greedy episode; no exploration
                print(f"\n  [inference @ {total_steps}] norm_ret {inf_ret:+.3f}")

                ckpt = ckpt_dir / f"dqn_{total_steps:06d}.pt"    # filename encodes the step count for easy sorting
                agent.save(str(ckpt))                             # save a timestamped checkpoint

                if inf_ret > best_return:                         # new best policy found
                    best_return = inf_ret                         # update the running best
                    agent.save(str(ckpt_dir / "best.pt"))         # overwrite best.pt with the improved policy
                    print(f"  new best: {best_return:.3f}  →  best.pt\n")

                next_eval += tr["eval_interval"]   # advance the trigger to the next evaluation milestone

    except KeyboardInterrupt:   # Ctrl+C pressed — exit gracefully rather than leaving the motor running
        print("\nInterrupted.")

    finally:                                            # always executed, even on exception
        env.estop()                                     # send emergency-stop so the motor halts immediately
        client.close()                                  # tear down ZMQ sockets cleanly
        agent.save(str(ckpt_dir / "final.pt"))          # preserve the last policy regardless of how training ended
        print("Saved final.pt")                         # confirm the final checkpoint was written


if __name__ == "__main__":
    main()   # entry point when the script is run directly: python train.py
