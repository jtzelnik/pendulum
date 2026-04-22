"""
Gymnasium-style environment wrapper around the live ZMQ connection to the LLI.

Observation vector (5-D, article eq. 4):
    [sin θ,  cos θ,  θ_dot,  x,  x_dot]
sin/cos are used instead of raw θ to avoid the angle discontinuity at ±π
that would look like a very large state jump to the network.

Actions (3 discrete integers):
    0 → left   duty = −DUTY
    1 → coast  duty =  0
    2 → right  duty = +DUTY

Reward (article eq. 11):
    r = (1/2)(1 − cos θ) − (x / x_max)²
    Maximum = 1 when θ = π (upright) and x = 0 (centred).
    An additional limit_penalty (−400) is applied on the terminal step
    when the proximity sensor was triggered (episode_status == 1), giving
    a minimum normalised-return of −400/800 = −0.5 as stated in the article.

Episode termination (either condition ends the episode):
    • episode_status != 0 — LLI detected a limit hit or |θ_dot| > 14 rad/s
    • step_count >= max_steps — maximum episode length reached
After a terminal step the LLI automatically runs its homing sequence; the
client does not need to command anything — env.reset() blocks until homing
is complete and packets resume.
"""

import math              # sin, cos for observation construction
import numpy as np       # float32 array returned as observation
from zmq_client import ZMQClient   # transport layer; recv_state / send_cmd


# ── PendulumEnv ───────────────────────────────────────────────────────────────
class PendulumEnv:
    """Real-hardware RL environment: one env step = one LLI control tick (20 ms).

    Mirrors the Gymnasium (gym) interface — reset() / step() — so it can be
    swapped for a simulation environment without changing the training loop.
    """

    N_ACTIONS = 3   # left / coast / right — matches the three motor commands in the article
    OBS_DIM   = 5   # [sin θ, cos θ, θ_dot, x, x_dot] — matches the ANN input layer size

    def __init__(self, client: ZMQClient, duty: int, x_max: float,
                 max_steps: int, limit_penalty: float) -> None:
        """Initialise the environment with hardware and episode parameters.

        Args:
            client:        Open ZMQClient connected to the Pi.
            duty:          PWM duty magnitude for left/right actions (0–255).
            x_max:         Track half-length in metres; used in reward denominator.
            max_steps:     Maximum steps per episode before forced termination.
            limit_penalty: Reward penalty added on a limit-sensor terminal step.
        """
        self._client        = client         # ZMQ transport; send_cmd / recv_state
        self._x_max         = x_max          # 0.35 m — track half-length used in reward and boundary check
        self._max_steps     = max_steps      # 800 steps — maximum episode length (article §RL environment)
        self._limit_penalty = limit_penalty  # −400 — subtracted from reward when proximity sensor fires
        self._actions       = [-duty, 0, duty]  # action index → signed PWM duty; index 0=left, 1=coast, 2=right
        self._step_count    = 0              # steps taken since the last reset; checked against max_steps

    # ── private helpers ───────────────────────────────────────────────────────

    def _obs(self, pkt) -> np.ndarray:
        """Convert a StatePacket into the 5-D observation vector fed to the DQN.

        Using sin/cos instead of θ directly avoids the −π/+π wraparound
        discontinuity, which would otherwise look like a huge state jump.
        """
        return np.array([           # build a contiguous float32 array for PyTorch
            math.sin(pkt.theta),    # sin θ — encodes angle without wraparound discontinuity
            math.cos(pkt.theta),    # cos θ — together with sin gives unique representation for all angles
            pkt.theta_dot,          # angular velocity in rad/s — already EMA-filtered by LLI
            pkt.x,                  # carriage position in metres (0 = centre)
            pkt.x_dot,              # carriage velocity in m/s — already EMA-filtered by LLI
        ], dtype=np.float32)        # float32 to match PyTorch default tensor dtype

    def _reward(self, pkt, terminal_status: int) -> float:
        """Compute the scalar reward for this transition (article eq. 11).

        r = (1/2)(1 − cos θ) − (x / x_max)²

        The angle term ranges from 0 (hanging, θ=0) to 1 (upright, θ=π).
        The position term penalises distance from centre; it equals 1 when
        |x| = x_max, so total reward at the boundary is at most 0.
        The limit_penalty is only added on the terminal step that triggered
        a proximity sensor — angular-velocity terminations are not penalised
        because the agent could not have avoided them so early in training.
        """
        r = 0.5 * (1.0 - math.cos(pkt.theta)) - (pkt.x / self._x_max) ** 2   # base reward: angle term − position penalty
        if terminal_status == 1:     # proximity sensor fired — cart hit a hard stop
            r += self._limit_penalty    # subtract 400 to strongly discourage driving into the rail ends
        return r   # scalar float; unbounded below (−400) and capped at 1.0

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Wait for the LLI to finish homing and return the initial observation.

        After a terminal step the LLI stops the motor, runs the full homing
        sequence (~120 s for the pendulum to settle), then resumes publishing
        at 50 Hz.  This method blocks until a packet with episode_status == 0
        is received, which signals that homing is complete and the new episode
        has started.
        """
        self._step_count = 0          # reset the per-episode step counter before the new episode
        self._client.flush()          # discard any stale packets that queued during homing
        while True:                   # spin until the LLI confirms the episode is running
            pkt = self._client.recv_state()        # blocking recv; returns immediately once packets resume
            if pkt.episode_status == 0:            # 0 = running — homing is finished and episode is live
                return self._obs(pkt)              # return the first observation of the new episode

    def step(self, action: int):
        """Send one motor command and receive the resulting state.

        Returns:
            obs:    Next 5-D observation vector.
            reward: Scalar reward for this transition.
            done:   True if the episode has ended (any terminal condition).
            info:   Dict with 'episode_status' for diagnostics.

        The LLI publishes a new StatePacket every 20 ms; recv_state() blocks
        until that packet arrives, so this method naturally runs at 50 Hz
        without any explicit sleep.
        """
        self._client.send_cmd(self._actions[action])   # translate action index (0/1/2) to signed duty and push to LLI
        pkt = self._client.recv_state()                # block ~20 ms for the next 50 Hz tick

        done = (                                         # episode ends if either condition is true:
            pkt.episode_status != 0                      # LLI flagged a limit hit (1) or angular-vel exceeded (2)
            or self._step_count >= self._max_steps - 1  # step budget exhausted (800 steps per article)
        )
        reward = self._reward(                           # compute scalar reward for this (s, a, s') transition
            pkt,
            pkt.episode_status if done else 0            # only apply limit_penalty on the actual terminal step
        )
        self._step_count += 1   # advance the per-episode counter before the next call

        return (
            self._obs(pkt),                             # next observation for the agent
            reward,                                     # scalar reward stored in replay buffer
            done,                                       # done flag breaks the episode collection loop
            {"episode_status": int(pkt.episode_status)},   # raw status code for logging / diagnostics
        )

    def estop(self) -> None:
        """Send an emergency-stop command — sets the estop flag in MotorCommand.

        Called from the training loop's finally block so the motor is always
        halted even if training is interrupted mid-episode.
        """
        self._client.send_cmd(0, estop=True)   # duty=0 and estop=1; LLI stops motor immediately on receipt
