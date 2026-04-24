"""
Gymnasium-style environment wrapper around the live ZeroMQ connection to the LLI.

This class bridges the RL training loop and the physical hardware. It translates
between the three-action discrete command space (left / coast / right) and the
raw signed PWM duties the LLI expects, and handles all the episode lifecycle
logic: detecting terminals, requesting homing, and waiting for the system to
be ready before each new episode.

Observation vector (5-D):
    [sin θ,  cos θ,  θ_dot,  x,  x_dot]

Why sin/cos instead of raw θ?
    θ is an angle, so it wraps around at ±π. If the pendulum passes through
    the upright position, θ might jump from +3.14 to -3.14 in one step. To
    the neural network that looks like an enormous state change, even though
    the physical change was tiny. Using (sin θ, cos θ) gives a smooth,
    unique representation for every angle without any discontinuity.

Actions (3 discrete integers):
    0 → duty = −DUTY  (drive left)
    1 → duty =  0     (coast — motor off)
    2 → duty = +DUTY  (drive right)

Reward (per step):
    r = (1/2)(1 − cos θ) − (x / x_max)²
    Range: 0 (hanging down, centred) to 1 (perfectly upright and centred).
    An additional limit_penalty (−400) is applied on the terminal step when
    the proximity sensor fires, discouraging the agent from driving into stops.

Episode termination:
    • episode_status != 0 — LLI detected a limit hit (1) or |θ_dot| > 14 rad/s (2).
      The LLI auto-homes in both cases, so reset() does not send request_home.
    • step_count >= max_steps — episode length cap reached (max_steps / loop_hz seconds).
      The LLI does NOT auto-home in this case, so reset() sends request_home.
"""

import math              # sin, cos for observation construction; used to encode angle without discontinuity
import time              # monotonic clock for request_home retry timeout
import numpy as np       # float32 array returned as observation (matches PyTorch default dtype)
from zmq_client import ZMQClient   # transport layer: recv_state() and send_cmd()
from protocol import (EPISODE_HOMING_STARTED,   # status code 3: LLI confirmed it received request_home
                      EPISODE_LIMIT_HIT,        # status code 1: carriage hit a rail stop; LLI auto-homes
                      EPISODE_ANGVEL_EXCEED)     # status code 2: |theta_dot| > 14 rad/s; LLI auto-homes


class PendulumEnv:
    """Real-hardware RL environment. One env step = one LLI control tick (1/loop_hz seconds)."""

    N_ACTIONS = 3   # number of discrete actions: left, coast, right
    OBS_DIM   = 5   # observation dimension: [sin θ, cos θ, θ_dot, x, x_dot]

    def __init__(self, client: ZMQClient, duty: int, x_max: float,
                 max_steps: int, limit_penalty: float, loop_hz: int) -> None:
        """Set up the environment with the open ZMQ connection and episode parameters.

        Args:
            client:        Open ZMQClient already connected to the Pi.
            duty:          PWM magnitude for left/right actions (0–255). Larger = faster.
            x_max:         Track half-length in metres (e.g. 0.35). Used in reward denominator.
            max_steps:     Maximum steps before the episode is force-terminated (e.g. 800).
            limit_penalty: Extra negative reward added when the carriage hits a limit (e.g. −400).
            loop_hz:       LLI control loop frequency in Hz; must match the compiled LLI binary.
        """
        self._client        = client
        self._x_max         = x_max
        self._max_steps     = max_steps
        self._limit_penalty = limit_penalty
        self._tick_ms       = int(1000 / loop_hz)   # one tick in milliseconds; scales poll/retry timeouts
        self._loop_hz       = loop_hz
        # Map action indices to signed duty values: 0=left, 1=coast, 2=right.
        self._actions       = [-duty, 0, duty]
        self._step_count           = 0     # steps taken in the current episode
        self._first_reset          = True  # True until the first reset() call; skips request_home because the LLI already homed at startup
        self._last_terminal_status = 0     # episode_status from the last terminal step (0 = max-steps, 1/2 = auto-homed)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _obs(self, pkt) -> np.ndarray:
        """Convert a StatePacket into the 5-D observation vector.

        Returns a float32 array because that is PyTorch's default tensor dtype,
        avoiding an implicit conversion every time the observation is passed to
        the neural network.
        """
        return np.array([
            math.sin(pkt.theta),    # sin θ — smooth angle encoding, no ±π discontinuity
            math.cos(pkt.theta),    # cos θ — together with sin gives a unique representation for all angles
            pkt.theta_dot,          # angular velocity in rad/s (Butterworth-filtered by LLI)
            pkt.x,                  # carriage position in metres (0 = centre)
            pkt.x_dot,              # carriage velocity in m/s (Butterworth-filtered by LLI)
        ], dtype=np.float32)

    def _reward(self, pkt, terminal_status: int) -> float:
        """Compute the scalar reward for one transition.

        Reward formula:
            r = (1/2)(1 − cos θ) − (x / x_max)²

        Angle term (1/2)(1 − cos θ):
            0 when hanging down (θ = 0), maximum 1 when upright (θ = π).
            Using cos θ rather than θ² gives a smooth, periodic signal.

        Position term (x / x_max)²:
            0 at the rail centre, 1 at either limit. Penalises drifting to the edges.

        The limit_penalty is only added when the proximity sensor fires (status=1),
        giving a strong signal to avoid the rail ends. Angular-velocity terminations
        (status=2) are not penalised because early in training the agent has no way
        to prevent them.
        """
        r = 0.5 * (1.0 - math.cos(pkt.theta)) - (pkt.x / self._x_max) ** 2
        if terminal_status == 1:
            r += self._limit_penalty   # strong negative signal for hitting the rail stop
        return r

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Wait for homing to finish and return the first observation of the new episode.

        Which of the three execution paths is taken depends on why the previous
        episode ended:

          Path A — First reset ever:
            The LLI ran a full homing sequence at startup before lli_loop started.
            We just confirm the LLI is publishing and return the first observation.
            No request_home is sent.

          Path B — Auto-home terminal (last status was 1 or 2, OR a status=1/2
            packet appeared in the gap between episodes):
            The LLI already stopped the motor and homed automatically. We wait for
            it to finish homing and resume publishing, then return.
            No request_home is sent.

          Path C — Max-steps terminal (last status was 0 and no gap auto-home):
            The LLI did not auto-home. We explicitly request homing and wait for
            the LLI to acknowledge before homing begins.
        """
        self._step_count = 0

        # Drain stale packets from the previous episode. Capture any auto-home
        # terminal (status=1/2) that the LLI published in the inter-episode gap
        # (e.g. while we were doing gradient updates). If the LLI auto-homed in
        # the gap but _last_terminal_status is still 0 (from the max-steps step),
        # pre_gap will be 1 or 2 and we will correctly take Path B.
        pre_gap = self._client.flush()

        # Phase 0: wait for the LLI to be live (blocks for up to 5 minutes to
        # accommodate the long startup homing sequence on the first episode).
        print("  [reset] waiting for LLI ...", flush=True)
        if not self._client.poll(300_000):
            raise RuntimeError("LLI not responding after 5 minutes — is the Pi running?")

        # Drain again: if the LLI just finished auto-homing, the status=1/2 packet
        # published before homing started may now be at the front of the queue.
        post_gap = self._client.flush()

        # Combine both flush results. Either flush catching a status=1 or 2 means
        # the LLI auto-homed since the last step and we should take Path B.
        gap_autohomed = pre_gap in (1, 2) or post_gap in (1, 2)

        if self._first_reset:
            # Path A: startup homing was already done before lli_loop started.
            self._first_reset = False
            print("  [reset] first reset — startup homing already complete", flush=True)

        elif self._last_terminal_status != 0 or gap_autohomed:
            # Path B: LLI already homed — either the terminal step said so, or we
            # saw a terminal status packet appear in the inter-episode gap.
            print(f"  [reset] auto-home path (last={self._last_terminal_status} gap={pre_gap}/{post_gap})", flush=True)

        else:
            # Path C: max-steps terminal with no auto-home — request it now.
            print("  [reset] requesting home ...", flush=True)
            self._request_home_and_wait_for_ack()

        # Phase 2: wait for the first status=0 packet, confirming homing is done
        # and the system is ready. During homing the LLI publishes nothing, so
        # recv_state() blocks here until homing finishes and publishing resumes.
        print("  [reset] homing in progress ...", flush=True)
        while True:
            pkt = self._client.recv_state()
            if pkt.episode_status == 0:
                return self._obs(pkt)   # homing complete — return first observation

    def _request_home_and_wait_for_ack(self) -> None:
        """Send request_home repeatedly until the LLI confirms homing has started.

        The LLI checks for request_home once per tick, so we retry every 2 ticks
        to ensure it is received even if a packet is missed.

        Two exit conditions — whichever arrives first:
          • The LLI sends episode_status=3 (HOMING_STARTED): explicit acknowledgement.
          • 4 ticks of silence (no packets at all): the LLI has entered homing(), which
            blocks its publish loop, confirming homing is running even without a status=3.

        We stop sending after 4 ticks of silence (at most 2 sends: t=0 and t=2 ticks)
        because continuing to send request_home after homing starts would put a new
        request into the command queue. The LLI's double-flush after homing is designed
        to drain exactly those 2 potentially queued messages; more would slip through.
        """
        deadline     = time.monotonic() + 30.0     # give up after 30 seconds
        last_send    = time.monotonic() - 1.0       # set to past so we send immediately on the first iteration
        last_packet_t = time.monotonic()            # tracks when we last received any packet

        while time.monotonic() < deadline:
            now = time.monotonic()

            if now - last_packet_t > 4.0 / self._loop_hz:
                # No packets for 4 ticks — homing is blocking the LLI's publish loop.
                # Stop sending and let Phase 2 wait for homing to finish.
                self._client.flush()
                return

            if now - last_send >= 2.0 / self._loop_hz:
                # Send (or re-send) the request_home command every 2 ticks.
                self._client.send_cmd(0, request_home=True)
                last_send = now

            if self._client.poll(self._tick_ms):   # wait up to one tick for a packet
                last_packet_t = time.monotonic()
                pkt = self._client.recv_state()
                if pkt.episode_status == EPISODE_HOMING_STARTED:
                    # LLI explicitly acknowledged — homing is now starting.
                    self._client.flush()   # discard any trailing packets before entering Phase 2
                    return
                if pkt.episode_status in (EPISODE_LIMIT_HIT, EPISODE_ANGVEL_EXCEED):
                    # The LLI hit a terminal condition at the same moment we sent
                    # request_home.  It will auto-home on its own — stop sending
                    # more commands immediately so the LLI's post-auto-home flush
                    # has as few stale request_home packets to drain as possible.
                    # Phase 2 (the while loop in reset()) will wait for status=0.
                    self._client.flush()
                    return

        raise RuntimeError("LLI did not acknowledge request_home within 30 s")

    def step(self, action: int):
        """Send one motor command and receive the resulting state (one control tick).

        The LLI publishes at loop_hz Hz, so recv_state() blocks for one tick period,
        which naturally paces the control loop without any explicit sleep.

        Args:
            action: Integer action index — 0 (left), 1 (coast), or 2 (right).

        Returns:
            obs:    Next observation vector (5-D float32 numpy array).
            reward: Scalar reward for this (state, action, next_state) transition.
            done:   True if the episode has ended (limit hit, ang-vel exceeded, or max steps).
            info:   Dict with 'episode_status' for logging and diagnostics.
        """
        self._client.send_cmd(self._actions[action])   # translate action index to signed duty and send
        pkt = self._client.recv_state()                # block until the LLI publishes the next state (~one tick)

        # Episode ends if the LLI flagged a safety condition OR we hit the step budget.
        done = (
            pkt.episode_status != 0                       # LLI flagged a limit hit (1) or ang-vel exceeded (2)
            or self._step_count >= self._max_steps - 1   # step budget exhausted
        )

        if done:
            # Record the terminal reason so reset() picks the correct homing path:
            #   0 = max-steps (LLI did not auto-home → reset() must request it)
            #   1 or 2 = auto-homed (reset() should wait, not request again)
            self._last_terminal_status = int(pkt.episode_status)

        # Compute reward; only apply the limit penalty on the actual terminal step.
        reward = self._reward(
            pkt,
            pkt.episode_status if done else 0   # pass 0 for non-terminal steps so penalty is never applied mid-episode
        )

        self._step_count += 1

        return (
            self._obs(pkt),
            reward,
            done,
            {"episode_status": int(pkt.episode_status)},
        )

    def estop(self) -> None:
        """Send an emergency-stop command to immediately halt the motor.

        Called from the training loop's finally block so the motor always
        stops even if training is interrupted by an exception or Ctrl+C.
        """
        self._client.send_cmd(0, estop=True)   # duty=0, estop=1 — LLI stops the motor on the next tick
