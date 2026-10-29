"""Select the DLL owner explicitly without duplicating integration/terrain code."""

from __future__ import annotations

import os

if os.environ.get("COSIM_SOLVER_BACKEND", "local") == "windows-rpc":
    from .solver_rpc import RemoteCarSim as CarSim
else:
    from .carsim import CarSim

__all__ = ["CarSim"]
