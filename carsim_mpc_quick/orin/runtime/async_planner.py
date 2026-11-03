"""Single-flight planner transport: high-rate control polls but never waits for inference."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor


class AsyncPlanner:
    """One in-flight request, immutable capture metadata, no fabricated fallback plans.

    The caller retains the last valid plan and determines its expiration from
    capture time, not arrival time. Transport failures propagate on poll.
    """

    def __init__(self, connection, model="qwen-drive-1.0-sft"):
        self.connection = connection
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-planner")
        self.future = None
        self.pending = None
        self.closed = False
        self.submitted = 0
        self.completed = 0
        self.skipped_busy = 0

    def submit(self, request):
        if self.closed:
            raise RuntimeError("Planner is closed")
        if self.future is not None:
            self.skipped_busy += 1
            return False
        packet = copy.deepcopy(request)
        packet["seq"] = self.submitted
        self.pending = packet
        self.sent_at = time.perf_counter()
        self.future = self.executor.submit(self.connection.exchange, packet)
        self.submitted += 1
        return True

    def poll(self, sim_time):
        if self.closed:
            raise RuntimeError("Planner is closed")
        if self.future is None or not self.future.done():
            return None
        reply, rpc_ms = self.future.result()
        request = self.pending
        if (
            reply.get("version") != 3
            or reply.get("seq") != request["seq"]
            or reply.get("model") != self.model
            or reply.get("captured_at") != request["captured_at"]
            or reply.get("origin") != request["origin"]
            or sim_time < request["captured_at"] - 1e-6
        ):
            raise ValueError("Asynchronous planner identity/capture mismatch")
        self.future = None
        self.pending = None
        self.completed += 1
        return dict(
            reply,
            rpc_ms=rpc_ms,
            received_sim_time=sim_time,
            arrival_age_s=sim_time - request["captured_at"],
            request_wall_seconds=time.perf_counter() - self.sent_at,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        # OrinConnection.close interrupts a blocked recv before the executor exits.
        self.connection.close()
        self.executor.shutdown(wait=True)
