"""
Deep Q-Network architecture (appendix S3).

Dense feed-forward network with two hidden layers of 256 nodes each,
ReLU activations, and a linear output layer:

    Input  (5)  →  Linear(5→256)  →  ReLU
                →  Linear(256→256) →  ReLU
                →  Linear(256→3)   →  Q-values (one per action)

The three output nodes emit raw (unbounded) Q-value estimates for the
three discrete actions: left / coast / right.  No softmax is applied —
the agent takes argmax over the outputs during action selection.

hidden_sizes is supplied from config.yaml so the depth and width can be
changed without touching this file.
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
        super().__init__()            # initialise nn.Module bookkeeping (parameter registration, etc.)
        layers: list = []             # accumulate layer objects before wrapping in Sequential
        in_size = obs_dim             # first layer's input width equals the observation dimension
        for h in hidden_sizes:                            # iterate over each hidden layer width
            layers += [nn.Linear(in_size, h), nn.ReLU()] # fully-connected layer followed by element-wise ReLU (appendix S3)
            in_size = h                                   # the next layer's input width equals this layer's output width
        layers.append(nn.Linear(in_size, n_actions))     # output layer: one Q-value per action, no activation (raw logits)
        self.net = nn.Sequential(*layers)                 # wrap the list into a single callable module for clean forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of observations through the network.

        Args:
            x: Float32 tensor of shape (batch, obs_dim).
        Returns:
            Q-value tensor of shape (batch, n_actions).
        """
        return self.net(x)   # delegate to the Sequential stack; output is (batch, n_actions) Q-values
