# Pendulum RL — PC-side control loop

Runs on the **client PC** (192.168.10.1) while the LLI runs on the **Pi** (192.168.10.2).
The Pi is the server; it binds both ZeroMQ sockets and the PC connects to them.

---

## Prerequisites

```
pip install torch pyzmq pyyaml numpy
```

The Pi must be running the LLI binary before you start either script.

---

## File layout

```
rl/
├── train.py        training entry point
├── evaluate.py     evaluation entry point
├── env.py          PendulumEnv — Gymnasium-style hardware wrapper
├── dqn.py          DQNAgent and ReplayBuffer
├── network.py      MLP Q-network (PyTorch)
├── zmq_client.py   ZeroMQ transport (SUB + PUSH sockets)
├── protocol.py     Binary wire format — StatePacket and MotorCommand
├── config.yaml     All hyperparameters and hardware settings
├── checkpoints/    Checkpoint .pt files saved during training
└── physical/       Best model from on-hardware training runs
```

---

## Network / wire protocol

Two TCP channels over the static ethernet link:

| Direction       | Socket pair       | Port | Content                          |
|-----------------|-------------------|------|----------------------------------|
| Pi → PC         | ZMQ_PUB / ZMQ_SUB | 5555 | `StatePacket` at 20 Hz (48 bytes) |
| PC → Pi         | ZMQ_PUSH / ZMQ_PULL | 5556 | `MotorCommand` (8 bytes)         |

**StatePacket** (48 bytes, little-endian):

| Field            | Type    | Description                                     |
|------------------|---------|-------------------------------------------------|
| `timestamp_us`   | int64   | Tick time (µs, monotonic clock)                 |
| `x`              | float64 | Carriage position in metres (0 = rail centre)   |
| `x_dot`          | float64 | Carriage velocity in m/s (Butterworth-filtered) |
| `theta`          | float64 | Pendulum angle in radians (0 = hanging down)    |
| `theta_dot`      | float64 | Angular velocity in rad/s (Butterworth-filtered)|
| `episode_status` | uint8   | 0 = running, 1 = limit hit, 2 = ang-vel exceeded |

**MotorCommand** (8 bytes, little-endian):

| Field   | Type   | Description                        |
|---------|--------|------------------------------------|
| `duty`  | int32  | Signed PWM duty (−255 to +255)     |
| `estop` | uint8  | 1 = immediate motor stop           |

---

## Observation, actions, and reward

**Observation vector** fed to the network (5-D):

```
[ sin θ,  cos θ,  θ_dot,  x,  x_dot ]
```

sin/cos are used instead of raw θ to avoid the ±π discontinuity.

**Actions** (3 discrete integers):

| Index | Command | Duty sent |
|-------|---------|-----------|
| 0     | Left    | −230      |
| 1     | Coast   | 0         |
| 2     | Right   | +230      |

**Reward** (per step):

```
r = 0.5 × (1 − cos θ) − (x / x_max)²
```

- Maximum +1.0 when pendulum is upright (θ = π) and carriage is centred (x = 0).
- An additional −400 penalty is added on the terminal step that triggers a proximity sensor.
- **Normalised return** = cumulative reward ÷ max_steps. Range: −0.5 (instant limit hit) to +1.0 (perfect full episode).

**Episode ends** when either:
- The LLI sets `episode_status != 0` (limit sensor or angular velocity > 14 rad/s), or
- 800 steps have elapsed.

After a terminal step the LLI automatically runs its homing sequence. The PC does not send any commands during homing — it just waits.

---

## Training

### How to run

```
cd hardware/rl
python train.py                                    # fresh run
python train.py --checkpoint checkpoints/best.pt   # resume from checkpoint
```

Press **Ctrl+C** at any time to abort. The motor is e-stopped and `final.pt` is saved before the process exits.

### What it does

```
config.yaml
    │
    ├─ ZMQClient          connects SUB→5555, PUSH→5556
    ├─ PendulumEnv        wraps ZMQClient with step() / reset()
    └─ DQNAgent           policy net, target net, replay buffer, optimiser

Outer loop  (until total_steps == 1 000 000)
│
├─ env.reset()
│   Flushes stale packets, then blocks until episode_status == 0.
│   The LLI homing sequence takes ~10–120 s depending on carriage position.
│
├─ Warmup  (first 2 400 steps)
│   Agent picks fully random actions to pre-populate the replay buffer
│   before gradient updates begin.
│
├─ Episode collection  (20 Hz, up to 800 steps)
│   Each tick:
│     1. Select action — ε-greedy (ε = 0.005) or random during warmup
│     2. send_cmd() → Pi via PUSH:5556
│     3. recv_state() ← Pi via SUB:5555  (blocks ~50 ms)
│     4. Compute reward and done flag
│     5. Store (s, a, r, s', done) in replay buffer
│     6. Every 1000 env steps — hard-copy policy net → target net
│
├─ End-of-episode training
│   If buffer has ≥ 1024 transitions:
│     Run one gradient step (Huber loss, Adam) per episode step.
│
└─ Inference checkpoint  (every 10 episodes)
    Run one fully greedy episode; print normalised return.
    Save  checkpoints/dqn_NNNNNN.pt
    If new best → overwrite  checkpoints/best.pt
```

Resuming from a checkpoint skips warmup and continues the step counter from where it left off.

### Terminal output

```
Connecting to Pi at 192.168.10.2:5555/5556  —  press Ctrl+C to abort

device: cpu
Training for 1000000 env steps.  Warmup 2400 steps (random).  Inference every 10 episodes.
  [reset] waiting for homing to complete ...
  ep    1  step  47/800  [R]  θ=+172.3°  x=+0.012m  r=+0.9321  Σ=  +43.81   ← live, updates every step
ep    1 | steps     47 | len  47 | norm_ret +0.055 | status 1                 ← printed when episode ends
         loss 0.0412

  [inference @ 5000] norm_ret +0.312
  new best: 0.312  →  best.pt
```

`status` values: `0` = clean end (max steps), `1` = limit sensor hit, `2` = angular velocity exceeded.

A per-episode CSV log is written to `checkpoints/train_YYYYMMDD_HHMMSS.csv` with columns:
`episode, total_steps, ep_len, norm_ret, mean_loss, episode_status`.

### Checkpoints

Saved in `checkpoints/` (created automatically):

| File                    | When saved                              |
|-------------------------|-----------------------------------------|
| `dqn_NNNNNN.pt`         | After every inference evaluation        |
| `best.pt`               | Overwritten each time a new best return is achieved |
| `final.pt`              | Always, on exit (normal or Ctrl+C)      |

Each `.pt` file contains `policy_net`, `target_net`, `optimizer` state, `total_steps`, and `episode`, so training can be resumed from any checkpoint via `--checkpoint`.

### DQN hyperparameters (from `config.yaml`)

| Parameter                | Value      |
|--------------------------|------------|
| Learning rate            | 0.000005   |
| ε (exploration)          | 0.005      |
| γ (discount)             | 0.995      |
| Warmup steps             | 2 400      |
| Replay buffer size       | 50 000     |
| Batch size               | 1 024      |
| Target net sync interval | 1 000 env steps |
| Hidden layers            | [256, 256] ReLU |

### Physical training results

The model in `physical/` was trained on the real hardware. Recent sessions reached ~0.78 normalised return at episode 1,071 (~846k env steps), with full 800-step episodes completing consistently.

---

## Evaluation

### How to run

```
cd hardware/rl
python evaluate.py checkpoints/best.pt
python evaluate.py checkpoints/best.pt --episodes 5
python evaluate.py checkpoints/best.pt --log-csv        # write per-step CSV
python evaluate.py onnx/cart_net.onnx                   # load from ONNX export
```

Press **Ctrl+C** at any time to abort. The motor is e-stopped before exit.

### What it does

```
config.yaml + checkpoint path
    │
    ├─ ZMQClient          same sockets as training
    ├─ PendulumEnv        same wrapper
    └─ DQNAgent           loaded from checkpoint; exploration disabled (greedy=True)

For each requested episode:
│
├─ env.reset()         same homing wait as training
│
└─ Episode rollout  (20 Hz, up to 800 steps)
    Each tick:
      1. Select action — always greedy (argmax Q)
      2. send_cmd() → Pi
      3. recv_state() ← Pi
      4. Accumulate reward

After all episodes: print mean ± std of normalised return.
```

No gradient updates are performed and the replay buffer is not used.

ONNX checkpoints are automatically converted to a `.pt` file on first load (saved alongside the `.onnx` file as `<name>_from_onnx.pt`).

### Terminal output

```
Connecting to Pi at 192.168.10.2:5555/5556  —  press Ctrl+C to abort

Loaded checkpoints/best.pt  (cpu)
  [reset] waiting for homing to complete ...
  ep   1  step 312/800  [·]  θ=+179.1°  dθ=+0.12r/s  x=-0.003m  dx=+0.01m/s  r=+0.9998  Σ= +312.41
Episode   1 | steps 800 | norm_ret +0.391 | status 0

Mean norm_ret over 1 episodes: +0.391  ±  0.000
```

### CSV logging (`--log-csv`)

When `--log-csv` is passed, a file `eval_YYYYMMDD_HHMMSS.csv` is written alongside the script with columns:

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| `episode`   | Episode number                                   |
| `step`      | Step within the episode                          |
| `x`         | Carriage position (m)                            |
| `x_dot`     | Carriage velocity (m/s)                          |
| `theta`     | Pendulum angle — unwrapped (rad)                 |
| `theta_dot` | Angular velocity (rad/s)                         |
| `command`   | Motor duty sent (−230, 0, or +230)               |
| `reward`    | Per-step reward                                  |

Theta is phase-unwrapped across timesteps so that a continuously spinning pendulum does not wrap at ±π.

---

## config.yaml reference

```yaml
loop_hz: 20          # control loop frequency; must match the compiled LLI binary (make LOOP_HZ=20)

connection:
  host:       "192.168.10.2"   # Pi static ethernet IP
  port_state: 5555             # Pi's PUB socket (state stream)
  port_cmd:   5556             # Pi's PULL socket (motor commands)

hardware:
  duty:  230      # PWM magnitude for left/right action (0–255)
  x_max: 0.35     # track half-length in metres

episode:
  max_steps:      800
  limit_penalty: -400.0

dqn:              # see DQN hyperparameters table above
  learning_rate:           0.000005
  epsilon:                 0.005
  gamma:                   0.995
  buffer_size:             50000
  batch_size:              1024
  target_update_interval:  1000

network:
  hidden_sizes: [256, 256]

training:
  max_steps:      1000000
  warmup_steps:   2400
  eval_interval:  10          # greedy inference episode every N training episodes
  checkpoint_dir: "checkpoints"
```
