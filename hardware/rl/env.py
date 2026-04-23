"""
Environment wrapper around the live ZMQ connection to the LLI.

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
For status != 0 terminals the LLI re-homes automatically. For max_steps
(status=0) terminals the caller must send request_home before calling
reset(), which then blocks until homing is complete and packets resume.
"""

import math              # sin, cos for observation construction
import time              # monotonic clock for request_home retry timeout
import numpy as np       # float32 array returned as observation
from zmq_client import ZMQClient   # transport layer; recv_state / send_cmd
from protocol import EPISODE_HOMING_STARTED   # status=3: LLI ack that homing is starting


# ── PendulumEnv ───────────────────────────────────────────────────────────────
class PendulumEnv:
    """Real-hardware RL environment: one env step = one LLI control tick (20 ms)."""

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
        self._step_count           = 0     # steps taken since the last reset; checked against max_steps
        self._first_reset          = True  # lli_main.cpp homes before lli_loop starts; skip request_home on the very first call
        self._last_terminal_status = 0     # episode_status of the last terminal step; 0=max-steps, 1/2=LLI auto-homed

    # ── private helpers ───────────────────────────────────────────────────────

    def _obs(self, pkt) -> np.ndarray:
        """Convert a StatePacket into the 5-D observation vector fed to the DQN.

        Using sin/cos instead of θ directly avoids the −π/+π wraparound
        discontinuity, which would otherwise look like a huge state jump.
        """
        return np.array([           # build a contiguous float32 array for PyTorch
            math.sin(pkt.theta),    # sin θ — encodes angle without wraparound discontinuity
            math.cos(pkt.theta),    # cos θ — together with sin gives unique representation for all angles
            pkt.theta_dot,          # angular velocity in rad/s — already Butterworth-filtered by LLI
            pkt.x,                  # carriage position in metres (0 = centre)
            pkt.x_dot,              # carriage velocity in m/s — already Butterworth-filtered by LLI
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
        """Wait for homing to complete (requesting it if needed) and return the first observation.

        Three execution paths depending on why the previous episode ended:

          Path A — First reset: lli_main.cpp completed startup homing before lli_loop
                   started. Phase 0 confirms lli_loop is publishing, then Phase 2
                   waits for status=0. No request_home is sent.

          Path B — Auto-home terminal (last status was 1 or 2): LLI stopped the motor
                   and began homing automatically. Phase 0 blocks until homing finishes
                   and the LLI resumes. Phase 2 then returns the first status=0 packet.
                   No request_home is sent.

          Path C — Max-steps terminal (last status was 0): LLI did not auto-home.
                   Phase 0 confirms the LLI is live, then Phase 1 sends request_home
                   and waits for the LLI's status=3 acknowledgement before homing
                   begins. Phase 2 waits for status=0.
        """
        self._step_count = 0
        # Drain packets that accumulated since the last step; capture any auto-home
        # terminal that the LLI may have published in the inter-episode gap.
        pre_gap = self._client.flush()

        # Phase 0: confirm lli_loop is publishing (5-min ceiling for startup homing)
        print("  [reset] waiting for LLI ...", flush=True)
        if not self._client.poll(300_000):
            raise RuntimeError("LLI not responding after 5 minutes — is the Pi running?")
        # Drain again; if the LLI just finished auto-homing this may hold the
        # status=1/2 packet that was published right before homing began.
        post_gap = self._client.flush()

        # If either flush saw a limit-hit (1) or ang-vel (2) terminal, the LLI
        # auto-homed in the gap — treat exactly like an auto-home terminal path.
        gap_autohomed = pre_gap in (1, 2) or post_gap in (1, 2)

        if self._first_reset:
            # Path A: startup homing already done — skip to Phase 2
            self._first_reset = False
            print("  [reset] first reset — startup homing already complete", flush=True)

        elif self._last_terminal_status != 0 or gap_autohomed:
            # Path B: LLI auto-homed (either the terminal step flagged it, or a
            # status=1/2 packet appeared in the inter-episode gap).
            print(f"  [reset] auto-home path (last={self._last_terminal_status} gap={pre_gap}/{post_gap})", flush=True)

        else:
            # Path C: max-steps terminal — LLI did not auto-home; request it explicitly.
            print("  [reset] requesting home ...", flush=True)
            self._request_home_and_wait_for_ack()

        # Phase 2: wait for status=0 — homing complete and system ready for next episode
        print("  [reset] homing in progress ...", flush=True)
        while True:
            pkt = self._client.recv_state()
            if pkt.episode_status == 0:
                return self._obs(pkt)

    def _request_home_and_wait_for_ack(self) -> None:
        """Send request_home until homing is confirmed running, then stop and return.

        Two exit conditions — whichever comes first:
          • status=3 received: LLI explicitly acknowledged the request.
          • 200 ms silence: LLI has gone quiet, meaning homing is blocking its loop.

        The silence path caps sends at 2 (t=0 and t=100 ms). The LLI's double
        flush after homing (50 ms sleep between flushes) clears both. Retransmitting
        beyond 200 ms of silence would put a new request_home into the freshly
        flushed queue and trigger a second homing.
        """
        deadline = time.monotonic() + 30.0
        last_send    = time.monotonic() - 1.0   # force immediate first send
        last_packet_t = time.monotonic()

        while time.monotonic() < deadline:
            now = time.monotonic()

            if now - last_packet_t > 0.2:
                # 200 ms with no packets — homing is definitely running.
                # Stop sending; Phase 2 will block until homing finishes.
                self._client.flush()
                return

            if now - last_send >= 0.1:
                self._client.send_cmd(0, request_home=True)
                last_send = now

            if self._client.poll(20):
                last_packet_t = time.monotonic()
                pkt = self._client.recv_state()
                if pkt.episode_status == EPISODE_HOMING_STARTED:
                    self._client.flush()
                    return

        raise RuntimeError("LLI did not acknowledge request_home within 30 s")

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
        if done:
            # Record termination reason so reset() knows which homing path to take:
            # non-zero means LLI auto-homed; 0 means max-steps and no auto-home.
            self._last_terminal_status = int(pkt.episode_status)
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
