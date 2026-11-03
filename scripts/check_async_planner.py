"""Verify nonblocking planner polling and immutable capture metadata, without simulators."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "carsim_mpc_quick/orin"))

from runtime.async_planner import AsyncPlanner  # noqa: E402


class DelayedConnection:
    def __init__(self, wrong_capture=False):
        self.release = threading.Event()
        self.started = threading.Event()
        self.wrong_capture = wrong_capture

    def exchange(self, request):
        self.started.set()
        if not self.release.wait(timeout=3):
            raise TimeoutError("Test transport was not released")
        return dict(
            version=3,
            seq=request["seq"],
            model="qwen-drive-1.0-sft",
            captured_at=request["captured_at"] + int(self.wrong_capture),
            origin=request["origin"],
            points=[],
        ), 100.0

    def close(self):
        self.release.set()


def main():
    for wrong in (False, True):
        connection = DelayedConnection(wrong)
        planner = AsyncPlanner(connection)
        request = dict(seq=99, captured_at=1.0, origin=[1, 2, 0])
        try:
            assert planner.submit(request)
            assert connection.started.wait(timeout=1)
            request["origin"][0] = 999
            assert not planner.submit(request)
            for _ in range(100):
                assert planner.poll(1.1) is None
            connection.release.set()
            deadline = time.monotonic() + 2
            while not planner.future.done():
                if time.monotonic() > deadline:
                    raise TimeoutError("Async worker did not finish")
                time.sleep(0.001)
            if wrong:
                try:
                    planner.poll(1.2)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Wrong capture timestamp accepted")
            else:
                reply = planner.poll(1.2)
                assert reply["origin"] == [1, 2, 0]
                assert reply["seq"] == 0
                assert abs(reply["arrival_age_s"] - 0.2) < 1e-9
                assert planner.poll(1.3) is None
        finally:
            planner.close()
    print("PASS: nonblocking poll, single flight, immutable capture, identity rejection")


if __name__ == "__main__":
    main()
