"""
ZeroMQ transport layer between the RL client (PC) and the LLI (Pi).

ZeroMQ is a messaging library that handles the low-level TCP connection,
buffering, and framing so we don't have to deal with raw sockets. It provides
two socket patterns here:

  SUB (subscriber) — receives StatePackets broadcast by the LLI's PUB socket.
    The LLI sends one packet per control tick; if the PC is slow it can fall behind.
    Calling flush() drains any queued-up packets before starting a new episode.

  PUSH — sends MotorCommands to the LLI's PULL socket.
    SNDHWM=1 means at most one command can queue inside ZeroMQ; if we send
    faster than the LLI reads, the oldest queued command is dropped. This
    prevents a build-up of stale commands from a previous tick.

The Pi is the "server" — it binds both sockets and waits for connections.
The PC is the "client" — it connects to the Pi's IP address and port numbers.
"""

import zmq                                          # pyzmq: Python bindings for the ZeroMQ C library
from protocol import StatePacket, unpack_state, pack_cmd   # wire-format helpers


class ZMQClient:
    """Manages the two ZeroMQ sockets used to communicate with the LLI.

    Create one instance before training starts; close it in the finally block.
    Both sockets are blocking by default — recv_state() will wait up to one tick
    for the next packet, which naturally paces the control loop.
    """

    def __init__(self, host: str, port_state: int, port_cmd: int) -> None:
        """Open the ZeroMQ context and connect both sockets to the Pi.

        Args:
            host:       IP address of the Raspberry Pi (e.g. '192.168.137.161').
            port_state: TCP port for receiving state packets (Pi's PUB socket, 5555).
            port_cmd:   TCP port for sending motor commands (Pi's PULL socket, 5556).
        """
        # A ZeroMQ context manages the background I/O threads. One per process.
        self._ctx = zmq.Context()

        # SUB socket: receives the broadcast state packets from the Pi.
        # setsockopt(SUBSCRIBE, b"") means "subscribe to all messages" —
        # ZeroMQ PUB/SUB supports topic filtering; empty prefix matches everything.
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(f"tcp://{host}:{port_state}")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")   # subscribe to all published messages

        # PUSH socket: sends motor commands to the Pi's PULL socket.
        # SNDHWM=1: if a command is already queued (e.g., we're sending faster than
        # the LLI reads), the new send raises zmq.error.Again rather than queueing
        # indefinitely. We catch that exception and drop the command — the next
        # tick's command will be fresher anyway.
        # LINGER=0: discard any unsent messages immediately when we close this socket
        # so shutdown doesn't stall waiting for the Pi to acknowledge.
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(f"tcp://{host}:{port_cmd}")
        self._push.setsockopt(zmq.SNDHWM, 1)   # drop commands rather than queue indefinitely
        self._push.setsockopt(zmq.LINGER, 0)   # don't wait for unsent messages on close

    def recv_state(self) -> StatePacket:
        """Block until the next StatePacket arrives and return it decoded.

        Normally returns within one control tick. During homing the LLI
        stops publishing, so this will block for the full duration of homing
        (~10–120 s depending on the sequence). That is expected behaviour —
        env.reset() calls recv_state() in a loop waiting for homing to finish.
        """
        return unpack_state(self._sub.recv())   # recv() blocks; returns raw bytes; unpack_state decodes to StatePacket

    def send_cmd(self, duty: int, estop: bool = False, request_home: bool = False) -> None:
        """Serialise and send a MotorCommand without blocking.

        If the send queue is full (SNDHWM=1 and a command is already waiting),
        the send is silently dropped. The LLI will receive the next tick's command.
        """
        try:
            self._push.send(pack_cmd(duty, estop, request_home), zmq.NOBLOCK)
        except zmq.error.Again:
            pass   # queue full — drop this command; next tick's command will be fresher

    def poll(self, timeout_ms: int) -> bool:
        """Return True if a StatePacket is available within timeout_ms, else False.

        Used by env.reset() to wait for the LLI to come online or finish homing
        without blocking indefinitely with no timeout.
        """
        return bool(self._sub.poll(timeout=timeout_ms))

    def flush(self) -> int:
        """Drain all queued StatePackets and return the highest episode_status seen.

        Called at the start of env.reset() to discard stale packets from the
        previous episode so the first observation of the new episode is fresh.

        Returns the maximum episode_status found while draining (0 if the queue
        was empty or all packets had status 0). A non-zero return value (1 or 2)
        means the LLI auto-homed in the gap between episodes — env.reset() uses
        this to avoid sending a redundant request_home.
        """
        max_status = 0
        while True:
            try:
                pkt = unpack_state(self._sub.recv(zmq.NOBLOCK))   # NOBLOCK: returns immediately if queue is empty
                if pkt.episode_status > max_status:
                    max_status = pkt.episode_status
            except zmq.error.Again:   # Again is raised when the queue is empty (equivalent to EAGAIN in C)
                break
        return max_status

    def close(self) -> None:
        """Tear down both sockets and the ZeroMQ context cleanly.

        Call this in the training loop's finally block to release all network
        resources even if training is interrupted by an exception.
        """
        self._sub.close()    # release the SUB socket
        self._push.close()   # release the PUSH socket (LINGER=0 means no waiting)
        self._ctx.term()     # shut down I/O threads and free the context
