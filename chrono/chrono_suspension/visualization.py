"""Native Irrlicht visualization, isolated from the simulation and RL workers.

Importing this module does not import PyChrono or create a window. Geometry is
rendered in a separate system of fixed, non-colliding bodies; it is never part of
the suspension dynamics. All coordinates use x forward, y left, z up, in metres.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

CORNER_NAMES = ("FL", "FR", "RL", "RR")
CAMERA_NAMES = ("iso", "side", "front", "top", "road")
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def result_path(value: str | Path) -> Path:
    """Resolve a generated artifact path, requiring the repository results folder."""
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError(f"Generated files must be inside {RESULTS_ROOT}, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_frame(frame: dict[str, Any]) -> None:
    """Validate the public JSONL render frame before passing values into C++."""
    vector_fields = {"body_pos": 3, "body_quat": 4, "road_m": 4}
    for name, size in vector_fields.items():
        value = frame.get(name, [])
        if len(value) != size or not all(math.isfinite(float(v)) for v in value):
            raise ValueError(f"Frame {name} must have {size} finite values")
    for name in ("wheel_pos", "suspension_top_pos", "suspension_bottom_pos"):
        if name != "wheel_pos" and name not in frame:
            continue
        values = frame.get(name, [])
        if len(values) != 4 or any(
            len(v) != 3 or not all(math.isfinite(float(x)) for x in v) for v in values
        ):
            raise ValueError(f"Frame {name} must be a finite 4 x 3 array")
    for name in ("actuator_force_n", "suspension_travel_m"):
        if name in frame and (
            len(frame[name]) != 4 or not all(math.isfinite(float(v)) for v in frame[name])
        ):
            raise ValueError(f"Frame {name} must have four finite values")
    if not math.isfinite(float(frame.get("time_s", float("nan")))):
        raise ValueError("Frame time_s must be finite")
    for name in ("roll_rad", "pitch_rad", "body_acc_mps2"):
        if name in frame and not math.isfinite(float(frame[name])):
            raise ValueError(f"Frame {name} must be a finite scalar")
    norm = sum(float(v) ** 2 for v in frame["body_quat"])
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("Frame body_quat must be a unit quaternion in w,x,y,z order")


class NativeViewer:
    """Render export_frame() dictionaries in a native Chrono Irrlicht window.

    Live viewing uses ``model.road`` without advancing or modifying the model.
    Replay passes ``None`` and an exactly reconstructed ``road``. Without a
    spatial road, legacy replay explicitly falls back to recorded contact pads.
    ``config`` is the fully resolved simulation config with an optional
    ``visualization`` mapping (width, height, cutaway, speed, camera).
    """

    def __init__(
        self,
        model: Any,
        config: dict[str, Any],
        *,
        road: Any = None,
        time_range_s: tuple[float, float] | None = None,
    ) -> None:
        try:
            self.chrono = importlib.import_module("pychrono")
            self.irr = importlib.import_module("pychrono.irrlicht")
        except ImportError as exc:
            raise RuntimeError(
                "Native viewing requires a PyChrono build containing pychrono.irrlicht. "
                "Activate the documented chrono-rl environment. Headless simulations "
                "do not create a renderer; the official Vehicle wrapper may still require "
                "the Irrlicht bindings to be installed."
            ) from exc
        self.config = config
        self.road = getattr(model, "road", None)
        if self.road is None:
            self.road = road
        self.terrain: Any = None
        if self.road is not None:
            from .terrain import RoadTerrain

            self.terrain = RoadTerrain(self.road, config)
        self._road_time_range = time_range_s or (0.0, config["physics"]["episode_seconds"])
        if (
            len(self._road_time_range) != 2
            or not all(math.isfinite(float(t)) for t in self._road_time_range)
            or self._road_time_range[0] > self._road_time_range[1]
        ):
            raise ValueError("time_range_s must be a finite (start, end) pair")
        self.road_body: Any = None
        self.road_mesh_stats: dict[str, Any] = {}
        self._road_meshes: list[tuple[Any, float]] = []
        self._road_status = (
            f"Spatial road: {self.road.kind} | same simulation input"
            if self.road is not None
            else "MISSING SPATIAL ROAD METADATA | recorded contact pads only"
        )
        visual = config.get("visualization", {})
        self.speed = float(visual.get("speed", 1.0))
        if not 0.05 <= self.speed <= 4.0:
            raise ValueError("Visualization speed must be between 0.05 and 4.0")
        self.paused = bool(visual.get("paused", False))
        self.cutaway = bool(visual.get("cutaway", True))
        self.closed = False
        self._camera_name = str(visual.get("camera", "iso"))
        self._last_cutaway = self.cutaway
        self._frame: dict[str, Any] | None = None
        self._shell_shapes: list[Any] = []
        self._gui_controls: dict[str, Any] = {}
        self._tk_panel: Any = None
        self._panel_vars: dict[str, Any] = {}
        self.system = self.chrono.ChSystemNSC()
        self.system.SetGravitationalAcceleration(self._vector((0, 0, 0)))
        self._build_scene()
        self.vis = self.irr.ChVisualSystemIrrlicht()
        self.vis.AttachSystem(self.system)
        self.vis.SetWindowSize(int(visual.get("width", 1280)), int(visual.get("height", 800)))
        self.vis.SetWindowTitle("Chrono | 7-DOF active suspension")
        self.vis.SetCameraVertical(self.chrono.CameraVerticalDir_Z)
        self.vis.Initialize()
        if self.vis.GetDevice() is None:
            raise RuntimeError("Irrlicht could not create a native window on this display")
        self.vis.AddCamera(self._vector((5, -5, 3)), self._vector((0, 0, 0.6)))
        self.vis.AddLightDirectional(
            65,
            45,
            self.chrono.ChColor(0.25, 0.25, 0.25),
            self.chrono.ChColor(0.08, 0.08, 0.08),
            self.chrono.ChColor(0.65, 0.65, 0.65),
        )
        self.vis.ShowInfoPanel(False)
        self.set_camera(self._camera_name)
        self._build_controls()

    def _vector(self, values: Any) -> Any:
        return self.chrono.ChVector3d(*(float(x) for x in values))

    def _body(self, position: Any = (0, 0, 0)) -> Any:
        body = self.chrono.ChBody()
        body.SetFixed(True)
        body.EnableCollision(False)
        body.SetPos(self._vector(position))
        self.system.AddBody(body)
        return body

    def _shape(
        self,
        body: Any,
        shape: Any,
        color: tuple[float, float, float],
        position: Any = (0, 0, 0),
        rotation: Any = None,
        opacity: float = 1.0,
    ) -> Any:
        material = self.chrono.ChVisualMaterial()
        material.SetDiffuseColor(self.chrono.ChColor(*color))
        material.SetAmbientColor(self.chrono.ChColor(*color))
        material.SetSpecularColor(self.chrono.ChColor(0.08, 0.08, 0.08))
        material.SetOpacity(opacity)
        shape.AddMaterial(material)
        frame = self.chrono.ChFramed(
            self._vector(position), rotation if rotation is not None else self.chrono.QUNIT
        )
        body.AddVisualShape(shape, frame)
        return shape

    def _box(self, body: Any, size: Any, color: Any, position: Any = (0, 0, 0)) -> Any:
        return self._shape(body, self.chrono.ChVisualShapeBox(*size), color, position)

    def _cabin(self, centre_x: float, half_width: float) -> Any:
        """Original low-poly cabin; cosmetic surfaces do not imply extra DOFs."""
        length = self._length * 0.52
        width = half_width * 1.65
        vertices = [
            (centre_x + sx * length * 0.5, sy * width * 0.5, 0.25)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ] + [
            (centre_x - 0.12 + sx * length * 0.31, sy * width * 0.42, 0.74)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        mesh = self.chrono.ChTriangleMeshConnected()
        faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        for a, b, c, d in faces:
            for triangle in ((a, b, c), (a, c, d)):
                mesh.AddTriangle(*(self._vector(vertices[i]) for i in triangle))
        shape = self.chrono.ChVisualShapeTriangleMesh()
        shape.SetMesh(mesh)
        return self._shape(self.chassis, shape, (0.24, 0.40, 0.50))

    def _build_road(self) -> None:
        """Build tiled static meshes once; rendering only translates their body.

        Station s is measured from the front axle at t=0: visual x=a+s-v*t.
        Lateral blending matches RoadTerrain: smoothstep inside +/-min(track),
        with flat same-side shoulders covering both front and rear wheel tracks.
        """
        visual, vehicle = self.config.get("visualization", {}), self.config["vehicle"]
        dx = float(visual.get("road_mesh_dx_m", 0.025))
        width = float(visual.get("road_width_m", 3.6))
        front = float(vehicle["front_distance_m"])
        rear = float(vehicle["rear_distance_m"])
        half_track = min(vehicle["front_half_track_m"], vehicle["rear_half_track_m"])
        if not math.isfinite(dx) or dx <= 0 or not math.isfinite(width):
            raise ValueError("Road mesh spacing must be positive and road width finite")
        if width / 2 <= max(vehicle["front_half_track_m"], vehicle["rear_half_track_m"]) + 0.2:
            raise ValueError("Road mesh must extend beyond both wheel tracks")
        if not math.isclose(float(self.road.wheelbase), front + rear, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Visual vehicle wheelbase differs from its resolved spatial road")
        road_speed = float(self.road.speed)
        start = road_speed * float(self._road_time_range[0]) - (front + rear) - 6.0
        end = road_speed * float(self._road_time_range[1]) + 16.0
        intervals = math.ceil((end - start) / dx)
        if intervals > 100_000:
            raise ValueError("Road mesh exceeds 100,000 stations; reduce viewed time range")
        dx = (end - start) / intervals
        # Seven transverse knots resolve the smooth interpolation between tracks.
        ys = [-width / 2, -half_track, -half_track / 2, 0, half_track / 2, half_track, width / 2]
        self.road_body = self._body()
        surface = markings = None
        previous = None
        triangles = 0
        minimum_height = math.inf
        maximum_height = -math.inf

        def point(s: float, y: float, lift: float = 0) -> Any:
            location = self._vector((front + s, y, 0))
            location.z = float(self.terrain.GetHeight(location)) + lift
            if not math.isfinite(location.z):
                raise ValueError("Terrain returned a nonfinite mesh height")
            return location

        def quad(mesh: Any, points: list[Any]) -> None:
            nonlocal triangles
            mesh.AddTriangle(points[0], points[1], points[2])
            mesh.AddTriangle(points[0], points[2], points[3])
            triangles += 2

        def attach(mesh: Any, color: tuple[float, float, float], lift: float = 0) -> None:
            shape = self.chrono.ChVisualShapeTriangleMesh()
            shape.SetMesh(mesh)
            self._shape(self.road_body, shape, color)
            self._road_meshes.append((mesh, lift))

        for index in range(intervals + 1):
            station = start + index * dx
            left, right = point(station, half_track).z, point(station, -half_track).z
            minimum_height = min(minimum_height, left, right)
            maximum_height = max(maximum_height, left, right)
            if previous is not None:
                if (index - 1) % 600 == 0:
                    surface = self.chrono.ChTriangleMeshConnected()
                    markings = self.chrono.ChTriangleMeshConnected()
                for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
                    quad(
                        surface,
                        [
                            point(previous, y0),
                            point(station, y0),
                            point(station, y1),
                            point(previous, y1),
                        ],
                    )
                # Paint sits 1 mm above the surface solely to avoid z-fighting.
                strips = [(-width / 2 + 0.12, 0.035), (width / 2 - 0.12, 0.035)]
                if ((previous + station) / 2) % 3.0 < 1.0:
                    strips.append((0.0, 0.025))
                for centre, half in strips:
                    quad(
                        markings,
                        [
                            point(previous, centre - half, 0.001),
                            point(station, centre - half, 0.001),
                            point(station, centre + half, 0.001),
                            point(previous, centre + half, 0.001),
                        ],
                    )
                if index % 600 == 0 or index == intervals:
                    attach(surface, (0.27, 0.31, 0.35))
                    attach(markings, (0.86, 0.89, 0.86), 0.001)
            previous = station
        self.road_mesh_stats = {
            "station_min_m": start,
            "station_max_m": end,
            "longitudinal_spacing_m": dx,
            "width_m": width,
            "triangles_including_markings": triangles,
            "minimum_track_height_m": minimum_height,
            "maximum_track_height_m": maximum_height,
            "frame": "vehicle fixed; road x = front_distance + station - speed*time",
        }

    def _build_scene(self) -> None:
        c = self.chrono
        vehicle = self.config["vehicle"]
        front = float(vehicle["front_distance_m"])
        rear = float(vehicle["rear_distance_m"])
        half_width = max(float(vehicle["front_half_track_m"]), float(vehicle["rear_half_track_m"]))
        self._length = front + rear + 1.0
        self._radius = float(vehicle["tire_radius_m"])
        centre_x = 0.5 * (front - rear)
        self.chassis = self._body()
        # A narrow frame preserves sight of all four suspension force elements.
        for y in (-0.43, 0.43):
            self._box(
                self.chassis, (front + rear, 0.08, 0.10), (0.20, 0.24, 0.29), (centre_x, y, -0.10)
            )
        for x in (front, -rear):
            self._box(self.chassis, (0.09, half_width * 2, 0.10), (0.27, 0.31, 0.36), (x, 0, -0.10))
        self._box(
            self.chassis,
            (front + rear - 0.3, 0.95, 0.15),
            (0.12, 0.40, 0.66),
            (centre_x, 0, 0.04),
        )
        shell = self._box(
            self.chassis,
            (self._length, 2 * half_width - 0.15, 0.24),
            (0.12, 0.40, 0.66),
            (centre_x, 0, 0.13),
        )
        cabin = self._cabin(centre_x - 0.15, half_width)
        self._shell_shapes.extend((shell, cabin))
        for shape in self._shell_shapes:
            shape.SetVisible(not self.cutaway)
        # Head lamps mark +X; red tail lamps mark -X, without implying a steerable model.
        for y in (-half_width * 0.65, half_width * 0.65):
            lamp = self._box(
                self.chassis, (0.05, 0.23, 0.08), (1, 0.9, 0.6), (front + 0.49, y, 0.12)
            )
            lamp.SetVisible(not self.cutaway)
            self._shell_shapes.append(lamp)
            lamp = self._box(
                self.chassis, (0.05, 0.23, 0.08), (0.85, 0.12, 0.10), (-rear - 0.49, y, 0.12)
            )
            lamp.SetVisible(not self.cutaway)
            self._shell_shapes.append(lamp)
        self.wheels = []
        self.road_pads = []
        for _ in CORNER_NAMES:
            wheel = self._body()
            # Chrono 10 cylinders have local Z axes; rotate axle into global Y.
            axle_rotation = c.QuatFromAngleX(math.pi / 2)
            self._shape(
                wheel,
                c.ChVisualShapeCylinder(self._radius, 0.23),
                (0.10, 0.11, 0.13),
                rotation=axle_rotation,
            )
            self._shape(
                wheel,
                c.ChVisualShapeCylinder(self._radius * 0.58, 0.24),
                (0.58, 0.65, 0.72),
                rotation=axle_rotation,
            )
            self.wheels.append(wheel)
            if self.road is None:
                pad = self._body()
                self._box(pad, (1.0, 0.70, 0.08), (0.27, 0.33, 0.40))
                self.road_pads.append(pad)
        if self.road is not None:
            self._build_road()
        else:
            floor = self._body((0, 0, -0.13))
            self._box(floor, (12, 8, 0.06), (0.72, 0.76, 0.80))

    def _build_controls(self) -> None:
        # Official Chrono 10 SWIG builds expose IGUIEnvironment but may return
        # opaque pointers for its widgets. Never call methods on those pointers.
        widget_types = ("IGUIStaticText", "IGUICheckBox", "IGUIScrollBar", "IGUIComboBox")
        if not all(hasattr(self.irr, name) for name in widget_types):
            self._build_tk_controls()
            return
        gui = self.vis.GetGUIEnvironment()
        rect = self.irr.recti
        self._hud = gui.addStaticText("", rect(15, 15, 460, 220), False, True)
        self._hud.setOverrideColor(self.irr.SColor(255, 20, 28, 36))
        self._gui_controls["pause"] = gui.addCheckBox(
            self.paused, rect(15, 225, 140, 250), None, -1, "Pause"
        )
        self._gui_controls["cutaway"] = gui.addCheckBox(
            self.cutaway, rect(150, 225, 300, 250), None, -1, "Cutaway"
        )
        gui.addStaticText("Playback speed: 0.05x .. 4.0x", rect(15, 257, 340, 277))
        speed = gui.addScrollBar(True, rect(15, 280, 300, 299))
        speed.setMin(5)
        speed.setMax(400)
        speed.setPos(int(round(self.speed * 100)))
        self._gui_controls["speed"] = speed
        camera = gui.addComboBox(rect(15, 310, 180, 335))
        for label in ("Isometric", "Side (pitch)", "Front (roll)", "Top", "Road ahead"):
            camera.addItem(label)
        camera.setSelected(CAMERA_NAMES.index(self._camera_name))
        self._gui_controls["camera"] = camera
        gui.addStaticText(
            "Mouse: rotate / pan / zoom\nGeometry scale: 1:1. Arrows: 0.15 m/kN\n"
            f"{self._road_status}\nNo visual collision forces; positive pitch: nose down",
            rect(15, 345, 430, 425),
            False,
            True,
        )

    def _build_tk_controls(self) -> None:
        """Use a native companion panel when SWIG does not expose GUI widgets."""
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            self.vis.Quit()
            raise RuntimeError(
                "This Irrlicht build omits GUI widget bindings. Install the conda tk package "
                "for the native control panel; headless training does not use it."
            ) from exc
        panel = tk.Tk()
        self._tk_panel = panel
        panel.title("Chrono | Suspension controls")
        panel.geometry("500x570+30+50")
        panel.minsize(500, 570)
        panel.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(panel, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="7-DOF ACTIVE SUSPENSION", font=("Segoe UI", 14, "bold")).pack(
            anchor="w", pady=(0, 14)
        )
        ttk.Label(
            frame,
            text=self._road_status,
            foreground="#1C5979" if self.road is not None else "#A51524",
            font=("Segoe UI", 10, "bold"),
            wraplength=460,
        ).pack(anchor="w", pady=(0, 12))
        variables = {
            "pause": tk.BooleanVar(value=self.paused),
            "cutaway": tk.BooleanVar(value=self.cutaway),
            "speed": tk.DoubleVar(value=self.speed),
            "camera": tk.StringVar(value=self._camera_name),
            "hud": tk.StringVar(value="Waiting for a simulation frame..."),
        }
        self._panel_vars = variables
        ttk.Label(frame, textvariable=variables["hud"], font=("Consolas", 10), justify="left").pack(
            anchor="w", pady=(0, 14)
        )
        toggles = ttk.Frame(frame)
        toggles.pack(fill="x", pady=5)
        ttk.Checkbutton(toggles, text="Pause", variable=variables["pause"]).pack(side="left")
        ttk.Checkbutton(toggles, text="Cutaway", variable=variables["cutaway"]).pack(
            side="left", padx=25
        )
        ttk.Label(frame, text="Playback speed: 0.05x .. 4.0x").pack(anchor="w", pady=(8, 0))
        ttk.Scale(frame, from_=0.05, to=4, variable=variables["speed"]).pack(fill="x")
        cameras = ttk.Frame(frame)
        cameras.pack(fill="x", pady=12)
        for name in CAMERA_NAMES:
            ttk.Radiobutton(
                cameras, text=name.title(), value=name, variable=variables["camera"]
            ).pack(side="left", padx=(0, 8))
        ttk.Label(
            frame,
            text="Mouse in 3D window: rotate / pan / zoom\n"
            "Geometry: 1:1. Force arrows: 0.15 m/kN\n"
            "Road geometry adds no visual collision forces.\n"
            "Road camera: see the surface moving past the car.\n"
            "Positive pitch means nose down.",
            justify="left",
        ).pack(anchor="w", pady=8)
        self.vis.GetGUIEnvironment().addStaticText(
            f"{self._road_status}\n"
            "Geometry 1:1 | Prescribed forward speed | Camera: iso / side / front / top / road",
            self.irr.recti(15, 15, 780, 55),
            False,
            True,
            None,
            -1,
            True,
        )
        panel.update_idletasks()

    def set_camera(self, name: str) -> None:
        cameras = {
            "iso": (2.9, -3.0, 1.8),
            "side": (0.0, -4.5, 1.3),
            "front": (3.5, 0.0, 1.3),
            "top": (0.0, 0.01, 4.5),
            "road": (-5.5, -5.5, 3.0),
        }
        if name not in cameras:
            raise ValueError(f"Unknown camera {name!r}; choose {tuple(cameras)}")
        self.vis.SetCameraPosition(self._vector(cameras[name]))
        self.vis.SetCameraTarget(self._vector((4.0, 0, 0.3) if name == "road" else (0, 0, 0.6)))
        self._camera_name = name

    def _suspension_points(self, frame: dict[str, Any]) -> tuple[Any, Any]:
        if "suspension_top_pos" in frame and "suspension_bottom_pos" in frame:
            return frame["suspension_top_pos"], frame["suspension_bottom_pos"]
        vehicle = self.config["vehicle"]
        corners = (
            (vehicle["front_distance_m"], vehicle["front_half_track_m"]),
            (vehicle["front_distance_m"], -vehicle["front_half_track_m"]),
            (-vehicle["rear_distance_m"], vehicle["rear_half_track_m"]),
            (-vehicle["rear_distance_m"], -vehicle["rear_half_track_m"]),
        )
        top = []
        for x, y in corners:
            point = self.chassis.TransformPointLocalToParent(self._vector((x, y, -0.10)))
            top.append((point.x, point.y, point.z))
        return top, frame["wheel_pos"]

    def render(self, frame: dict[str, Any]) -> bool:
        """Render one frame and process native widgets; return False after close."""
        if self.closed:
            return False
        validate_frame(frame)
        if not self.vis.Run():
            self.close()
            return False
        if self._tk_panel is not None:
            self._tk_panel.update()
            if self.closed:
                return False
            controls = self._panel_vars
            self.paused = bool(controls["pause"].get())
            self.speed = float(controls["speed"].get())
            self.cutaway = bool(controls["cutaway"].get())
            camera = str(controls["camera"].get())
        else:
            controls = self._gui_controls
            self.paused = bool(controls["pause"].isChecked())
            self.speed = float(controls["speed"].getPos()) / 100
            self.cutaway = bool(controls["cutaway"].isChecked())
            camera = CAMERA_NAMES[controls["camera"].getSelected()]
        if self.cutaway != self._last_cutaway:
            for shape in self._shell_shapes:
                shape.SetVisible(not self.cutaway)
            self.vis.UnbindItem(self.chassis)
            self.vis.BindItem(self.chassis)
            self._last_cutaway = self.cutaway
        if camera != self._camera_name:
            self.set_camera(camera)
        c = self.chrono
        self.chassis.SetPos(self._vector(frame["body_pos"]))
        self.chassis.SetRot(c.ChQuaterniond(*(float(v) for v in frame["body_quat"])))
        for index, wheel in enumerate(self.wheels):
            position = frame["wheel_pos"][index]
            wheel.SetPos(self._vector(position))
            if self.road_pads:
                self.road_pads[index].SetPos(
                    self._vector((position[0], position[1], float(frame["road_m"][index]) - 0.04))
                )
        if self.road_body is not None:
            self.road_body.SetPos(self._vector((-self.road.speed * frame["time_s"], 0, 0)))
        self.system.Update(float(frame["time_s"]))
        forces = frame.get("actuator_force_n", [0.0] * 4)
        travel = frame.get("suspension_travel_m", [0.0] * 4)
        accel = frame.get("body_acc_mps2", 0.0)
        if isinstance(accel, (list, tuple)):
            accel = accel[-1]
        hud = (
            f"CHRONO  |  7-DOF ACTIVE SUSPENSION\n"
            f"t = {float(frame['time_s']):7.3f} s    speed {self.speed:.2f}x\n"
            f"body az = {float(accel):+.3f} m/s2\n"
            f"roll = {math.degrees(float(frame.get('roll_rad', 0))):+.3f} deg   "
            f"pitch = {math.degrees(float(frame.get('pitch_rad', 0))):+.3f} deg\n"
            "corner    travel [mm]   actuator [N]\n"
        )
        if self.road is not None:
            hud = hud.replace(
                "corner    travel [mm]   actuator [N]",
                f"road speed = {self.road.speed:.2f} m/s; "
                f"station = {self.road.speed * frame['time_s']:.2f} m\n"
                "corner    travel [mm]   actuator [N]",
            )
        hud += "\n".join(
            f"{name:>3}       {1000 * float(travel[i]):+8.2f}       {float(forces[i]):+8.1f}"
            for i, name in enumerate(CORNER_NAMES)
        )
        if self._tk_panel is not None:
            self._panel_vars["hud"].set(hud)
        else:
            self._hud.setText(hud)
        self.vis.BeginScene()
        self.vis.Render()
        top, bottom = self._suspension_points(frame)
        for i in range(4):
            a, b = self._vector(top[i]), self._vector(bottom[i])
            self.irr.drawSpring(self.vis, 0.07, a, b, c.ChColor(0.9, 0.57, 0.14), 72, 7)
            force = float(forces[i])
            self.irr.drawLabel3D(
                self.vis,
                f"{CORNER_NAMES[i]}  {force:+.0f} N",
                a + self._vector((0, 0.25 if i % 2 == 0 else -0.25, 0.30)),
                c.ChColor(1.0, 0.96, 0.82),
            )
            if abs(force) > 1:
                origin = a + self._vector((0, 0.18 if i % 2 == 0 else -0.18, 0))
                direction = a - b
                direction.Normalize()
                end = origin + direction * (force * 0.00015)
                color = c.ChColor(0.9, 0.2, 0.12) if force >= 0 else c.ChColor(0.1, 0.4, 0.9)
                arrow_direction = direction * (1 if force >= 0 else -1)
                head_size = min(0.04, abs(force) * 0.00015 * 0.30)
                side = c.VECT_Y.Cross(arrow_direction)
                side.Normalize()
                head_base = end - arrow_direction * head_size
                self.irr.drawSegment(self.vis, origin, end, color)
                for sign in (-1, 1):
                    self.irr.drawSegment(
                        self.vis, end, head_base + side * (sign * head_size * 0.6), color
                    )
        self.vis.EndScene()
        self._frame = frame
        return True

    def screenshot(self, path: str | Path) -> Path:
        """Capture the last frame to PNG/JPEG within repository results/."""
        if self.closed or self._frame is None:
            raise RuntimeError("Render a frame before requesting a screenshot")
        target = result_path(path)
        if target.exists():
            raise FileExistsError(f"Screenshot already exists: {target}")
        self.vis.WriteImageToFile(str(target))
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Irrlicht did not write screenshot: {target}")
        return target

    def close(self) -> None:
        if not self.closed:
            self.vis.Quit()
            self.closed = True
        if self._tk_panel is not None:
            self._tk_panel.destroy()
            self._tk_panel = None
