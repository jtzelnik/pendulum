"""
DQN agent: experience-replay buffer, ε-greedy policy, and the Bellman update.

All techniques are taken directly from the article's appendix S2 and Table 2:

  Replay buffer       — 50 000 transitions; breaks temporal correlation between
                        consecutive samples that would bias gradient estimates.
  Fixed Q-targets     — a second 'target' network provides stable Bellman
                        bootstrap values; synced from the policy net every
                        C = 1000 environment steps (called from train.py).
  Huber loss          — equivalent to clipping the TD-error gradient to [−1, 1],
                        preventing large errors from destabilising early training.
  ε-greedy            — fixed ε = 0.178 throughout training (no decay for DQN;
                        contrast with the decaying schedule used for Q-learning).
  Adam optimiser      — PyTorch defaults (β₁=0.9, β₂=0.999, ε=1e-8).
  Train frequency     — one gradient step per environment step, applied in a
                        batch at the end of each episode (train.py drives this).
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
        random action; otherwise pick the action with the highest Q-value.
        During evaluation (greedy=True): always pick the best action.

        Args:
            obs:    Current observation as a float32 numpy array (shape (5,)).
            greedy: If True, suppress exploration and always act greedily.
        Returns:
            Integer action index in {0, 1, 2}.
        """
        if not greedy and random.random() < self.epsilon:           # ε-greedy: explore with probability ε
            return random.randrange(self.N_ACTIONS)                 # uniform random action to encourage state-space coverage
        state = torch.tensor(obs, dtype=torch.float32,              # wrap numpy obs in a tensor
                             device=self.device).unsqueeze(0)       # add batch dimension: (5,) → (1, 5)
        with torch.no_grad():                                        # no gradient needed for inference
            return int(self.policy_net(state).argmax(dim=1).item()) # forward pass → Q-values → argmax over actions → Python int

    def train_step(self) -> float:
        """Sample one mini-batch from the buffer and perform one gradient update.

        Implements the DQN Bellman update (appendix eq. 2):
            target = r + γ · max_{a'} Q(s', a'; w⁻)  · (1 − done)
            loss   = HuberLoss(Q(s, a; w), target)

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

    def save(self, path: str) -> None:
        """Persist network weights and optimiser state to a .pt checkpoint file.

        Both networks and the optimiser are saved so training can be resumed
        exactly from this point, including Adam's moment estimates.

        Args:
            path: Filesystem path for the checkpoint, e.g. 'checkpoints/best.pt'.
        """
        torch.save({                                              # save a dict so keys are explicit and forward-compatible
            "policy_net": self.policy_net.state_dict(),          # trained network weights
            "target_net": self.target_net.state_dict(),          # frozen target weights
            "optimizer":  self.optimizer.state_dict(),           # Adam moment estimates and step count
        }, path)                                                  # write to disk

    def load(self, path: str) -> None:
        """Restore network weights and optimiser state from a checkpoint file.

        map_location ensures a GPU-saved checkpoint can be loaded on CPU and
        vice versa without manual device re-mapping.

        Args:
            path: Path to the .pt checkpoint file written by save().
        """
        ckpt = torch.load(path, map_location=self.device)           # load checkpoint; remap tensors to self.device
        self.policy_net.load_state_dict(ckpt["policy_net"])          # restore trained weights
        self.target_net.load_state_dict(ckpt["target_net"])          # restore target weights
        self.optimizer.load_state_dict(ckpt["optimizer"])            # restore Adam state so training continues smoothly
