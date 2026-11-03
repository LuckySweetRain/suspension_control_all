"""Process-local VehicleSim DLL lifecycle and unchanged 8/77 + pose I/O mapping."""

from __future__ import annotations

import ctypes as ct
import math
import os
import re

from .paths import CARSIM_LIBRARY, CARSIM_PROGRAM, CARSIM_TEMPLATE, DATABASE


class CarSim:
    """One DLL owner; keep original 8 imports / 77 exports and append 4 pose exports."""

    def __init__(self, output, terrain=None, contact=False, driving=False, stop_time=None):
        if os.name != "nt" and CARSIM_LIBRARY.suffix.lower() == ".dll":
            raise RuntimeError(
                "Linux cannot load a Windows CarSim DLL. Use scripts/run_cosim.py "
                "from WSL to launch the Windows frontend, or supply a licensed Linux solver."
            )
        output = output.resolve()
        text = CARSIM_TEMPLATE.read_text(encoding="utf-8")
        if stop_time is not None:
            if not math.isfinite(stop_time) or stop_time <= 0:
                raise ValueError("Invalid CarSim stop time")
            configured = re.search(r"(?m)^TSTOP\s+(\S+)", text)
            if configured is None:
                raise ValueError("Missing CarSim stop time")
            text = re.sub(
                r"(?m)^TSTOP\s+\S+", f"TSTOP {max(stop_time, float(configured[1]))}", text
            )
        if terrain is not None:
            road = re.search(
                r"(?ms)^ENTER_PARSFILE Roads\\3D_Road[^\n]*\n.*?^EXIT_PARSFILE Roads\\3D_Road[^\n]*\.par[ \t]*$",
                text,
            )
            if road is None:
                raise RuntimeError("Expected one expanded CarSim road block")
            path = re.search(
                r"(?ms)^ENTER_PARSFILE Roads\\BuilderSegment[^\n]*\n.*?^EXIT_PARSFILE Roads\\BuilderSegment[^\n]*",
                road.group(),
            )
            if path is None:
                raise RuntimeError("Missing independent driver path")
            text = text[: road.start()] + path.group() + text[road.end() :]
        self.contact = contact
        self.driving = driving
        extra_imports = ["IMPORT IMP_FX_AERO Add 0.0"] if contact else []
        if driving:
            # Units verified in vehicle_imp.txt generated from this solver/model:
            # steering-wheel degrees, normalized throttle, four actuator pressures MPa.
            extra_imports += [
                "IMPORT IMP_STEER_SW Replace 0.0",
                "IMPORT IMP_THROTTLE_ENGINE Replace 0.0",
            ] + ["IMPORT IMP_PBK_" + wheel + " Replace 0.0" for wheel in ("L1", "L2", "R1", "R2")]
        self.n_import = 8 + len(extra_imports)
        if extra_imports:
            text = re.sub(r"(?m)^PORTS_IMP\s+1,8\s*$", f"PORTS_IMP 1,{self.n_import}", text)
            text = re.sub(r"(?m)^END\s*$", "\n".join(extra_imports) + "\nEND", text)
        self.n_export = 82 if driving else 81
        text = re.sub(r"(?m)^PORTS_EXP\s+1,77\s*$", f"PORTS_EXP 1,{self.n_export}", text)
        text = re.sub(
            r"(?m)^END\s*$",
            "EXPORT Xcg_TM\nEXPORT Ycg_TM\nEXPORT Yaw\nEXPORT Roll\n"
            + ("EXPORT Vy_SM\n" if driving else "")
            + "END",
            text,
        )
        par = output / "runtime.par"
        par.write_text(text, encoding="utf-8")
        sim = output / "runtime.sim"
        prefix = output / "LastRun"
        dll = str(CARSIM_LIBRARY)
        library_keyword = "DLLFILE" if os.name == "nt" else "SOFILE"
        program_directory = CARSIM_PROGRAM if os.name == "nt" else CARSIM_LIBRARY.parent
        sim.write_text(
            f"SIMFILE\nINPUT {par}\nECHO {prefix}_echo.par\nLOGFILE {prefix}_log.txt\n"
            f"FINAL {prefix}_end.par\nERDFILE {prefix}.vs\nPROGDIR {program_directory}/\n"
            f"DATADIR {DATABASE}/\nVEHICLE_CODE i_i\nEXT_MODEL_STEP 0.001\n"
            f"PORTS_IMP 1,{self.n_import}\nPORTS_EXP 1,{self.n_export}\n{library_keyword} {dll}\nEND\n",
            encoding="utf-8",
        )
        self.dll = ct.CDLL(dll)
        self.dll.vs_set_opt_error_dialog(ct.c_int(0))
        self.dll.vs_get_error_message.restype = ct.c_char_p
        self.terrain = terrain
        if terrain is not None:
            terrain.install(self.dll)
        self.dll.vs_read_configuration.argtypes = [
            ct.c_char_p,
            ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.dll.vs_integrate_io.argtypes = [
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.dll.vs_copy_export_vars.argtypes = [ct.POINTER(ct.c_double)]
        self.dll.vs_terminate_run.argtypes = [ct.c_double]
        ni, ne = ct.c_int32(), ct.c_int64()
        start, stop, dt = ct.c_double(), ct.c_double(), ct.c_double()
        old = os.getcwd()
        try:
            os.chdir(str(DATABASE))
            self.dll.vs_read_configuration(
                str(sim).encode(),
                ct.byref(ni),
                ct.byref(ne),
                ct.byref(start),
                ct.byref(stop),
                ct.byref(dt),
            )
        finally:
            os.chdir(old)
        self.t = start.value
        self.check()
        if (ni.value, ne.value) != (self.n_import, self.n_export) or abs(dt.value - 0.001) > 1e-10:
            self.close()
            raise RuntimeError(f"Unexpected CarSim I/O or step: {(ni.value, ne.value, dt.value)}")
        self.values = (ct.c_double * self.n_export)()
        self.dll.vs_copy_export_vars(self.values)
        self.t = start.value
        self.initial = list(self.values)
        if terrain is not None:
            if not terrain.origin_calibrated:
                terrain.origin_cs = (self.initial[77], self.initial[78])
                terrain.cache.clear()
            terrain.install(self.dll)

    def check(self):
        if self.terrain is not None:
            self.terrain.check()
        if self.dll.vs_error_occurred():
            raise RuntimeError(self.dll.vs_get_error_message().decode(errors="replace"))

    def step(self, forces, contact_force=0, driving_inputs=None):
        """Advance 10 ms; optional drive inputs are (steering wheel deg, throttle, MPa).

        This low-level boundary deliberately takes solver units. Road-wheel angle
        conversion belongs to a calibrated vehicle adapter, not this DLL wrapper.
        """
        values = list(forces)
        if len(values) != 8 or not all(math.isfinite(v) for v in values):
            raise ValueError("Expected eight finite suspension inputs")
        if self.contact:
            if not math.isfinite(contact_force):
                raise ValueError("Nonfinite contact force")
            values.append(contact_force)
        if self.driving:
            if driving_inputs is None or len(driving_inputs) != 3:
                raise ValueError("Driving mode requires (steering wheel deg, throttle, brake MPa)")
            steer, throttle, brake = driving_inputs
            if not all(math.isfinite(v) for v in driving_inputs):
                raise ValueError("Nonfinite driving inputs")
            if abs(steer) > 540 or not 0 <= throttle <= 1 or not 0 <= brake <= 15:
                raise ValueError("Driving inputs exceed integration limits")
            values.extend([steer, throttle, brake, brake, brake, brake])
        elif driving_inputs is not None:
            raise ValueError("Driving inputs supplied to a suspension-only run")
        inputs = (ct.c_double * self.n_import)(*values)
        self.last_substeps = []
        for _ in range(10):
            self.t += 0.001
            status = self.dll.vs_integrate_io(self.t, inputs, self.values)
            self.check()
            if status:
                raise RuntimeError(f"CarSim integration stopped: {status} at {self.t}")
            if getattr(self, "capture_substeps", False):
                sample = list(self.values)
                if not all(math.isfinite(v) for v in sample):
                    raise RuntimeError("Nonfinite CarSim integration substep")
                self.last_substeps.append(dict(time=self.t, values=sample))
        if not all(math.isfinite(v) for v in self.values):
            raise RuntimeError("Nonfinite CarSim state")
        return list(self.values)

    def close(self):
        self.dll.vs_terminate_run(self.t)
