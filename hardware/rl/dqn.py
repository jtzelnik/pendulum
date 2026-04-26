"""
DQN agent: experience-replay buffer, ε-greedy policy, and the Bellman update.

Key ideas explained for beginners:

  Replay buffer       — Stores the last 50 000 (state, action, reward,
                        next_state, done) tuples.  Training samples a random
                        mini-batch from this buffer rather than using the most
                        recent transitions in order.  This matters because
                        consecutive environment steps are highly correlated
                        (each step follows naturally from the last), and
                        training on correlated data causes the network to
                        overfit to the current situation and forget the rest.
                        Random sampling mixes old and recent experience so the
                        gradient estimate covers the whole state space.

  Fixed Q-targets     — Two identical networks are kept: the 'policy net'
                        (updated every gradient step) and the 'target net'
                        (frozen and only copied from the policy net every 1000
                        env steps).  The Bellman target is computed from the
                        frozen target net.  Without this, the prediction and
                        the target would both change every step — like trying
                        to hit a bullseye that moves every time you shoot.

  Huber loss          — Similar to mean-squared error (MSE) for small errors
                        but switches to mean-absolute error (MAE) for large
                        ones.  This clips the gradient magnitude so that a
                        single very wrong prediction at the start of training
                        cannot send the network weights flying off to infinity.

  ε-greedy            — With probability ε (0.178) the agent picks a random
                        action instead of the best one.  This forces the agent
                        to visit states it would not choose on its own, which
                        is necessary for learning — if you only do what you
                        already think is best, you never discover better paths.

  Adam optimiser      — Adaptive gradient descent: automatically adjusts the
                        learning rate for each weight based on past gradient
                        magnitudes (β₁=0.9, β₂=0.999).

  Train frequency     — One gradient step is run per environment step, but
                        batched at the end of each episode (train.py drives this).
"""

import random                           # uniform random sampling for ε-greedy and buffer
import numpy as np                      # array stacking when building sample batches
import torch                            # tensor ops, device management
import torch.nn as nn                   # not used directly but kept for consistency with network.py
import torch.nn.functional as F         # F.huber_loss — Huber / smooth-L1 loss
import torch.optim as optim             # Adam optimiser
from collections import deque           # O(1) append and pop; maxlen auto-evicts oldest entries
from typing import List, Tuple          # type hints

from network import DQN                 # the MLP Q-function defined in network.py


# ── ReplayBuffer ──────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) transitions.

    When the buffer is full, the oldest transition is silently discarded to
    make room for the new one (deque maxlen behaviour).  Sampling is uniform
    random, which breaks the temporal correlation between consecutive steps
    that would bias the gradient if transitions were used in order.
    """

    def __init__(self, capacity: int) -> None:
        """Allocate the buffer with the given maximum number of transitions.

        Args:
            capacity: Maximum number of transitions to store (50 000).
        """
        self._buf: deque = deque(maxlen=capacity)   # deque with automatic eviction of oldest element when full

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Append one transition to the buffer.

        Args:
            state:      Observation before the action (float32, shape (5,)).
            action:     Integer action index taken (0, 1, or 2).
            reward:     Scalar reward received after the action.
            next_state: Observation after the action (float32, shape (5,)).
            done:       True if this transition ended the episode.
        """
        self._buf.append((state, action, reward, next_state, done))   # store as a plain tuple; deque evicts oldest when full

    def sample(self, batch_size: int) -> Tuple:
        """Draw a uniform random mini-batch of transitions.

        Returns five tensors (states, actions, rewards, next_states, dones)
        ready to be moved to the training device.  np.array() is called before
        torch.tensor() to avoid a slow element-wise copy of a Python list.

        Args:
            batch_size: Number of transitions to sample (1024).
        """
        batch = random.sample(self._buf, batch_size)                          # uniform random sample without replacement
        states, actions, rewards, next_states, dones = zip(*batch)            # transpose list-of-tuples into five separate tuples
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),   # (batch, 5) observation matrix
            torch.tensor(actions,               dtype=torch.long),       # (batch,)  action indices for gather()
            torch.tensor(rewards,               dtype=torch.float32),    # (batch,)  scalar rewards
            torch.tensor(np.array(next_states), dtype=torch.float32),   # (batch, 5) next-observation matrix
            torch.tensor(dones,                 dtype=torch.float32),    # (batch,)  0.0 or 1.0; multiplied into Bellman target
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buf)   # used by train.py to guard against training before the buffer has enough data


# ── DQNAgent ──────────────────────────────────────────────────────────────────
class DQNAgent:
    """Full DQN agent: policy network, target network, replay buffer, and training.

    The separation between policy_net and target_net is the 'fixed Q-targets'
    trick: the Bellman target uses target_net weights (w⁻) while the gradient
    is applied to policy_net weights (w).  This decouples the prediction from
    the target, preventing the feedback loop that causes divergence in naïve
    Q-learning with function approximation.

    update_target() is called from train.py (not from train_step()) so that
    the sync happens on environment-step boundaries rather than gradient-step
    boundaries, which is what the paper specifies.
    """

    OBS_DIM   = 5   # [sin θ, cos θ, θ_dot, x, x_dot] — fixed by the problem definition
    N_ACTIONS = 3   # left / coast / right — fixed by the problem definition

    def __init__(self, hidden_sizes: List[int], lr: float, epsilon: float,
                 gamma: float, buffer_size: int, batch_size: int,
                 target_update_interval: int, device: str = "cpu") -> None:
        """Construct both networks, the optimiser, and the replay buffer.

        Args:
            hidden_sizes:           Layer widths for the MLP, e.g. [256, 256].
            lr:                     Adam learning rate (0.0003).
            epsilon:                Fixed ε-greedy exploration rate (0.178).
            gamma:                  Discount factor γ for the Bellman target (0.995).
            buffer_size:            Replay buffer capacity (50 000).
            batch_size:             Mini-batch size for each gradient step (1024).
            target_update_interval: Env steps between target-net syncs (1000).
            device:                 'cpu' or 'cuda'; tensors are moved here.
        """
        self.epsilon    = epsilon          # fixed exploration rate — no decay schedule for DQN (appendix Table 2)
        self.gamma      = gamma            # discount factor — 0.995 weights future rewards heavily
        self.batch_size = batch_size       # mini-batch size — 1024 per appendix Table 2
        self.device     = torch.device(device)   # torch.device object used when moving tensors

        self.policy_net = DQN(self.OBS_DIM, self.N_ACTIONS, hidden_sizes).to(self.device)   # trained network: weights w updated each gradient step
        self.target_net = DQN(self.OBS_DIM, self.N_ACTIONS, hidden_sizes).to(self.device)   # frozen copy: weights w⁻ provide stable Bellman targets
        self.target_net.load_state_dict(self.policy_net.state_dict())   # initialise target identical to policy so early targets are not random
        self.target_net.eval()   # put target in eval mode: disables dropout/batchnorm updates (not used here, but good practice)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)   # Adam with default β₁=0.9, β₂=0.999 as specified in appendix
        self.buffer    = ReplayBuffer(buffer_size)   # circular replay buffer; train.py calls buffer.add() each env step

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Choose an action using ε-greedy policy.

        During training (greedy=False): with probability ε pick a uniformly
        random action; otherwise ask the network for the best action.
        Random exploration is essential early in training — if the agent only
        does what it currently thinks is best, it never visits unfamiliar states
        and the Q-values for those states never improve.

        During evaluation (greedy=True): always pick the highest-Q action so
        the reported return is a clean measure of policy quality.

        Args:
            obs:    Current observation as a float32 numpy array (shape (5,)).
            greedy: If True, suppress exploration and always act greedily.
        Returns:
            Integer action index in {0, 1, 2}.
        """
        if not greedy and random.random() < self.epsilon:           # explore: probability ε → random action
            return random.randrange(self.N_ACTIONS)                 # uniform random: all three actions equally likely
        state = torch.tensor(obs, dtype=torch.float32,              # convert numpy array to a PyTorch tensor
                             device=self.device).unsqueeze(0)       # add batch dimension: shape (5,) → (1, 5)
        with torch.no_grad():                                        # disable gradient tracking — we're only reading Q-values
            return int(self.policy_net(state).argmax(dim=1).item()) # forward pass → Q-values for all 3 actions → pick the largest

    def train_step(self) -> float:
        """Sample one mini-batch from the buffer and perform one gradient update.

        What we are trying to teach the network:
            For every (state, action, reward, next_state) tuple in the batch,
            the network's Q-value for (state, action) should satisfy:

                Q(s, a) = reward + γ · best_Q(next_state)

            This is the Bellman equation.  It says: 'the value of being in
            state s and taking action a equals the immediate reward PLUS the
            discounted value of the best option in the next state.'  The
            network is trained to satisfy this equation simultaneously for all
            sampled transitions.  If Q(s,a) is too large or too small, the
            gradient of the Huber loss pushes it toward the Bellman target.

        Why the target net (w⁻) instead of the policy net (w) for the target?
            If we computed the Bellman target with the same network we are
            training, both sides of the equation would move on every gradient
            step — the network would chasing its own tail and diverge.
            The target net is frozen between syncs (every 1000 env steps),
            giving the policy net a stable goal to fit toward.

        Bellman update (canonical form):
            target = r + γ · max_{a'} Q(s', a'; w⁻) · (1 − done)
            loss   = HuberLoss(Q(s, a; w), target)
            The (1 − done) term zeros out the future term on terminal steps
            because there is no next state to bootstrap from.

        Returns 0.0 without updating if the buffer has fewer entries than
        batch_size — training only starts once the buffer is sufficiently full.

        Returns:
            Scalar loss value for logging; 0.0 if skipped.
        """
        if len(self.buffer) < self.batch_size:    # guard: don't train on a nearly empty buffer — estimates would be too noisy
            return 0.0                            # signal to caller that no update was performed

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)   # draw random mini-batch
        states      = states.to(self.device)       # move to GPU if available
        actions     = actions.to(self.device)      # move to GPU if available
        rewards     = rewards.to(self.device)      # move to GPU if available
        next_states = next_states.to(self.device)  # move to GPU if available
        dones       = dones.to(self.device)        # move to GPU if available; 1.0 on terminal transitions

        # Q(s, a; w) — current network's estimate for the taken action only
        q_values = (self.policy_net(states)               # forward pass: (batch, 3) Q-values for all actions
                    .gather(1, actions.unsqueeze(1))       # select the Q-value of the action actually taken
                    .squeeze(1))                           # remove the action dimension: (batch, 1) → (batch,)

        with torch.no_grad():                                              # target computation has no gradient — only policy_net is trained
            next_q  = self.target_net(next_states).max(dim=1).values      # max Q(s', a'; w⁻) over actions — Bellman bootstrap from frozen net
            targets = rewards + self.gamma * next_q * (1.0 - dones)       # Bellman target; (1-done) zeros out future term on terminal steps

        # Huber loss is equivalent to clipping the TD-error gradient to [−1, 1]
        # per appendix S2; delta=1.0 is the PyTorch default
        loss = F.huber_loss(q_values, targets)   # mean Huber loss over the batch

        self.optimizer.zero_grad()   # clear gradients from the previous step — PyTorch accumulates by default
        loss.backward()              # backpropagate through policy_net; target_net has no grad so it is unaffected
        self.optimizer.step()        # apply Adam update to policy_net parameters

        return loss.item()   # return scalar loss for the training log

    def update_target(self) -> None:
        """Copy policy-network weights into the target network (hard update).

        Called by train.py every target_update_interval environment steps,
        not every gradient step — this matches the paper's specification that
        C is measured in time steps, not optimizer steps.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())   # overwrite all target-net parameters with current policy-net values

    def save(self, path: str, **extras) -> None:
        """Persist network weights, optimiser state, and any extra scalars.

        Callers can pass keyword arguments (e.g. total_steps=50000, episode=200)
        that are stored alongside the network weights so training can resume from
        exactly the same point.

        Args:
            path:    Filesystem path for the checkpoint, e.g. 'checkpoints/best.pt'.
            **extras: Arbitrary scalar values to embed in the checkpoint dict.
        """
        torch.save({
            "policy_net":    self.policy_net.state_dict(),   # trained network weights
            "target_net":    self.target_net.state_dict(),   # frozen target weights
            "optimizer":     self.optimizer.state_dict(),    # Adam moment estimates and step count
            "replay_buffer": list(self.buffer._buf),         # full transition history for seamless resume
            **extras,                                        # e.g. total_steps, episode
        }, path)

    def load(self, path: str) -> dict:
        """Restore network weights and optimiser state from a checkpoint file.

        Returns a dict of any extra values that were saved alongside the
        networks (e.g. {"total_steps": 50000, "episode": 200}).  Old
        checkpoints without extras return an empty dict.

        map_location ensures a GPU-saved checkpoint loads correctly on CPU
        and vice versa.

        Checkpoints converted from ONNX (via load_onnx()) have no optimizer
        key; in that case Adam starts fresh from its default initial state.

        Args:
            path: Path to the .pt checkpoint file written by save() or load_onnx().
        Returns:
            Dict of non-network keys stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])   # restore trained weights
        self.target_net.load_state_dict(ckpt["target_net"])   # restore target weights
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])   # restore Adam state so training continues smoothly
        else:
            print("[load] no optimizer state in checkpoint — Adam starts fresh")
        if "replay_buffer" in ckpt:
            self.buffer._buf.extend(ckpt["replay_buffer"])       # repopulate buffer so training resumes without a cold-start
            print(f"[load] restored {len(self.buffer)} transitions into replay buffer")
        return {k: v for k, v in ckpt.items()
                if k not in ("policy_net", "target_net", "optimizer", "replay_buffer")}   # return training-loop scalars to caller

    @staticmethod
    def load_onnx(onnx_path: str, out_pt_path: str, hidden_sizes: List[int]) -> None:
        """Convert an ONNX weight file into a .pt checkpoint compatible with load().

        Extracts network weights from the ONNX graph and packages them into the
        same dict format that save() writes.  Both policy_net and target_net are
        set to the extracted weights.  Optimizer state is absent — training
        resumes with a fresh Adam optimiser.

        Parameter matching (tried in order):
          1. By name — works when the ONNX was exported from a PyTorch DQN with
             the same architecture; initializer names equal the state_dict keys
             (e.g. 'net.0.weight', 'net.2.bias').
          2. By shape/position — iterates ONNX initializers in graph order and
             maps each to the corresponding expected parameter by shape.  Handles
             external tools that assign generic names.

        Args:
            onnx_path:    Path to the source .onnx file.
            out_pt_path:  Path to write the converted .pt checkpoint.
            hidden_sizes: MLP hidden layer widths (must match the ONNX model,
                          e.g. [256, 256]).
        Raises:
            ImportError:  if the 'onnx' package is not installed.
            ValueError:   if the number of matched weight tensors does not match
                          the expected parameter count for the given hidden_sizes.
        """
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            raise ImportError(
                "pip install onnx   (required to load .onnx checkpoints)"
            )

        # Build expected parameter names and shapes from a reference model instance.
        ref = DQN(DQNAgent.OBS_DIM, DQNAgent.N_ACTIONS, hidden_sizes)
        expected = [(name, tuple(p.shape)) for name, p in ref.named_parameters()]
        # e.g. [('net.0.weight', (256, 5)), ('net.0.bias', (256,)), ...]

        onnx_model = onnx.load(onnx_path)

        # Attempt 1: match by parameter name.
        by_name = {init.name: numpy_helper.to_array(init)
                   for init in onnx_model.graph.initializer}
        if all(name in by_name for name, _ in expected):
            state_dict = {
                name: torch.tensor(by_name[name].copy())
                for name, _ in expected
            }
            print("[onnx→pt] matched weights by parameter name")
        else:
            # Attempt 2: filter initializers whose shape (or its transpose) appears
            # in the expected set, then match positionally in graph order.  Some
            # exporters (TF, JAX, stable-baselines3) store Linear weights as
            # (in, out) rather than PyTorch's (out, in), so we accept both and
            # transpose on load when needed.
            expected_shapes = {shape for _, shape in expected}
            expected_shapes_T = {shape[::-1] for shape in expected_shapes if len(shape) == 2}

            candidates = [
                init for init in onnx_model.graph.initializer
                if tuple(init.dims) in expected_shapes or tuple(init.dims) in expected_shapes_T
            ]
            if len(candidates) != len(expected):
                raise ValueError(
                    f"Expected {len(expected)} weight tensors for "
                    f"hidden_sizes={hidden_sizes}, "
                    f"found {len(candidates)} matching shapes in '{onnx_path}'.\n"
                    f"Confirm the ONNX architecture is "
                    f"input={DQNAgent.OBS_DIM} → {hidden_sizes} → "
                    f"output={DQNAgent.N_ACTIONS}."
                )
            state_dict = {}
            for (name, exp_shape), init in zip(expected, candidates):
                t = torch.tensor(numpy_helper.to_array(init).copy())
                if t.shape != torch.Size(exp_shape):
                    t = t.T   # stored transposed — flip to (out, in)
                state_dict[name] = t
            print("[onnx→pt] matched weights by shape/position (ONNX names differed)")

        torch.save({"policy_net": state_dict, "target_net": state_dict}, out_pt_path)
        print(f"[onnx→pt] saved → {out_pt_path}")
