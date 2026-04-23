"""
Deep Q-Network (DQN) — the neural network that learns to estimate Q-values.

What is a Q-value?
    Q(s, a) is the total reward the agent expects to collect for the rest of
    the episode if it takes action a in state s and then acts optimally from
    that point on.  The agent picks the action with the highest Q-value.

Why a neural network instead of a lookup table?
    The state space is continuous (5 floats), so there are infinitely many
    possible states — a table is impossible.  The network learns to approximate
    Q(s, a) for any state by generalising from the (state, reward) examples it
    has seen.  Training adjusts the weights until Q(s, a) accurately predicts
    long-term reward.

Architecture:
    Input (5) → Linear(5→256) → ReLU
              → Linear(256→256) → ReLU
              → Linear(256→3)   — one Q-value per action, no output activation

Why no softmax on the output?
    Softmax turns numbers into probabilities that sum to 1.  Q-values are
    expected cumulative rewards — they can be any real number and do not need
    to sum to anything.  The agent takes argmax (pick the action with the
    largest value), so softmax would distort the comparison.

hidden_sizes is passed in from config.yaml so the depth and width can be
changed without editing this file.
"""

import torch                # tensor operations and autograd
import torch.nn as nn       # Module base class, Linear, ReLU, Sequential
from typing import List     # type hint for hidden_sizes parameter


# ── DQN ───────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    """Parametric Q-function Q(s, a; θ) implemented as a dense MLP.

    Instantiated twice by DQNAgent: once as the policy network (trained each
    step) and once as the target network (synced every C environment steps).
    The two-network design breaks the feedback loop between the prediction
    and the bootstrap target, stabilising training (appendix S2).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int]) -> None:
        """Build the sequential layer stack from the supplied dimensions.

        Args:
            obs_dim:      Number of input features (5 for this problem).
            n_actions:    Number of discrete actions (3: left/coast/right).
            hidden_sizes: List of hidden-layer widths, e.g. [256, 256].
        """
        super().__init__()            # nn.Module bookkeeping: registers parameters so optimiser can find them
        layers: list = []             # accumulate layer objects in a plain list before wrapping
        in_size = obs_dim             # first layer's input width = observation size
        for h in hidden_sizes:                              # one loop iteration per hidden layer
            layers += [nn.Linear(in_size, h), nn.ReLU()]   # Linear: weighted sum of inputs + bias; ReLU: replace negatives with 0
            in_size = h                                     # this layer's output is the next layer's input
        layers.append(nn.Linear(in_size, n_actions))       # output layer: one raw Q-value per action, no activation
        self.net = nn.Sequential(*layers)                   # Sequential calls each layer in order on every forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of observations through the network and return Q-values.

        PyTorch calls this automatically when you write `net(x)` — you should
        never call `net.forward(x)` directly.

        Args:
            x: Float32 tensor of shape (batch, obs_dim).  Each row is one
               observation; batch=1 during single-step action selection,
               batch=1024 during a training step.
        Returns:
            Float32 tensor of shape (batch, n_actions) — one Q-value per
            action per observation.  The caller uses .argmax(dim=1) to pick
            the best action, or .gather(1, actions) to select specific ones.
        """
        return self.net(x)   # pass x through Linear→ReLU→Linear→ReLU→Linear in sequence
