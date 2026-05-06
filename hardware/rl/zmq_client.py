"""
ZeroMQ transport layer between the RL client (PC) and the LLI (Pi).

The Pi is the server — it binds both sockets.  The PC connects to those
bound addresses.  Socket roles mirror the LLI:
  LLI  ZMQ_PUB  binds  tcp://*:5555   →  PC  ZMQ_SUB  connects
  LLI  ZMQ_PULL binds  tcp://*:5556   →  PC  ZMQ_PUSH connects
"""

import zmq                                          # pyzmq — Python bindings for the ZeroMQ C library
from protocol import StatePacket, unpack_state, pack_cmd   # wire-format helpers defined in protocol.py


# ── ZMQClient ─────────────────────────────────────────────────────────────────
class ZMQClient:
    """Open and manage the two ZeroMQ sockets used to talk to the LLI.

    Lifetime: construct once before the training loop, call close() in the
    finally block.  Both sockets are blocking by default; recv_state() will
    block until the next 20 Hz packet arrives (~50 ms), which naturally
    paces the control loop.
    """

    def __init__(self, host: str, port_state: int, port_cmd: int) -> None:
        """Create context and open SUB + PUSH sockets.

        Args:
            host:       IP address of the Pi (e.g. '192.168.10.2').
            port_state: TCP port the LLI's PUB socket is bound to (5555).
            port_cmd:   TCP port the LLI's PULL socket is bound to (5556).
        """
        self._ctx = zmq.Context()   # one ZeroMQ context per process; owns the I/O threads

        self._sub = self._ctx.socket(zmq.SUB)                         # SUB receives from the LLI's PUB broadcast
        self._sub.connect(f"tcp://{host}:{port_state}")               # connect to Pi's bound PUB address
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")                      # empty prefix = subscribe to all messages (no topic filter)

        self._push = self._ctx.socket(zmq.PUSH)                       # PUSH feeds into the LLI's PULL queue
        self._push.connect(f"tcp://{host}:{port_cmd}")                # connect to Pi's bound PULL address
        self._push.setsockopt(zmq.SNDHWM, 1)                         # high-water mark of 1: drop oldest command if queue is full, preventing stale commands building up
        self._push.setsockopt(zmq.LINGER, 0)                         # discard unsent messages immediately on close, so shutdown is not delayed

    def recv_state(self) -> StatePacket:
        """Block until the next StatePacket arrives and return it decoded.

        The 20 Hz LLI tick means this normally returns within ~50 ms.
        During homing, no packets are sent, so this will block for up to
        ~120 s — that is expected and handled by env.reset().
        """
        return unpack_state(self._sub.recv())   # recv() blocks; returns raw bytes; unpack_state decodes to StatePacket

    def send_cmd(self, duty: int, estop: bool = False, request_home: bool = False) -> None:
        """Send one MotorCommand to the LLI without blocking."""
        try:
            self._push.send(pack_cmd(duty, estop, request_home), zmq.NOBLOCK)
        except zmq.error.Again:
            pass   # HWM reached — drop this command; the Pi will get the next one

    def poll(self, timeout_ms: int) -> bool:
        """Return True if a StatePacket is available within timeout_ms, else False."""
        return bool(self._sub.poll(timeout=timeout_ms))

    def flush(self) -> int:
        """Drain all stale StatePackets and return the highest episode_status seen.

        Called at the start of env.reset() so the first observation the
        agent sees is fresh, not a packet from the previous episode.
        Returns the maximum episode_status found while draining (0 if the
        queue was empty or all packets had status 0). A non-zero return (1 or
        2) means the LLI auto-homed in the gap — env.reset() uses this to
        avoid sending a redundant request_home.
        """
        max_status = 0
        while True:
            try:
                pkt = unpack_state(self._sub.recv(zmq.NOBLOCK))   # NOBLOCK: returns immediately if no message is waiting
                if pkt.episode_status > max_status:
                    max_status = pkt.episode_status
            except zmq.error.Again:            # Again is raised when the queue is empty (EAGAIN)
                break                          # queue is drained — exit loop
        return max_status

    def close(self) -> None:
        """Tear down sockets and the ZeroMQ context cleanly."""
        self._sub.close()    # release the SUB socket and its file descriptors
        self._push.close()   # release the PUSH socket; LINGER=0 means no waiting
        self._ctx.term()     # block until all sockets are closed, then free the context
