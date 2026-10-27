"""Deterministic longitudinal test roads rendered as real CARLA OpenDRIVE meshes.

Profiles are lane-wide: a half-car trough is not a localized left/right pothole.
Analytic heights generate the mesh and are used for independent verification only;
the controller still receives depth images and tyres query the CARLA scene mesh.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

LENGTH_M = 160.0
SPAWN_STATION_M = 10.0
LANE_CENTER_Y_M = 2.0  # CARLA reflects OpenDRIVE's lateral coordinate.
PROFILES = ("wave", "bump", "pothole", "combined")


def elevation(station, profile):
    z = 0.0
    if profile in ("wave", "combined") and 25 <= station <= 50:
        distance = station - 25
        edge = min(distance / 2.5, (25 - distance) / 2.5, 1.0)
        envelope = edge**3 * (10 - 15 * edge + 6 * edge**2)
        z += 0.015 * envelope * math.sin(2 * math.pi * distance / 5)
    for kind, start, height in [("bump", 60.0, 0.030), ("pothole", 85.0, -0.025)]:
        if profile in (kind, "combined") and start <= station <= start + 4:
            z += height * 0.5 * (1 - math.cos(2 * math.pi * (station - start) / 4))
    return z


def slope(station, profile):
    delta = 1e-4
    return (elevation(station + delta, profile) - elevation(station - delta, profile)) / (2 * delta)


def specification(profile):
    return dict(
        profile=profile,
        enabled_features=list(PROFILES[:3]) if profile == "combined" else [profile],
        length_m=LENGTH_M,
        spawn_station_m=SPAWN_STATION_M,
        lane_width_m=4.0,
        wave=dict(start_m=25, end_m=50, amplitude_m=0.015, wavelength_m=5.0),
        bump=dict(start_m=60, length_m=4.0, height_m=0.030),
        pothole=dict(start_m=85, length_m=4.0, depth_m=0.025, geometry="lane_wide_trough"),
        elevation_knot_spacing_m=0.1,
        mesh_vertex_distance_m=0.05,
    )


def opendrive(profile, parallel=False):
    if profile not in PROFILES:
        raise ValueError("Unknown rough-road profile")
    root = ET.Element("OpenDRIVE")
    ET.SubElement(
        root,
        "header",
        revMajor="1",
        revMinor="4",
        name="Suspension_" + profile,
        version="1.00",
        date="2026-10-27",
        north="160",
        south="0",
        east="4",
        west="-4",
    )
    ET.SubElement(
        root.find("header"), "geoReference"
    ).text = "+proj=tmerc +lat_0=42 +lon_0=2 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    road = ET.SubElement(
        root, "road", name="Suspension test road", length=str(LENGTH_M), id="1", junction="-1"
    )
    ET.SubElement(road, "link")
    road_type = ET.SubElement(road, "type", s="0", type="rural")
    ET.SubElement(road_type, "speed", max="35", unit="km/h")
    plan = ET.SubElement(road, "planView")
    geometry = ET.SubElement(plan, "geometry", s="0", x="0", y="0", hdg="0", length=str(LENGTH_M))
    ET.SubElement(geometry, "line")
    elevations = ET.SubElement(road, "elevationProfile")
    step = 0.1
    for index in range(round(LENGTH_M / step)):
        s = index * step
        z0, z1 = elevation(s, profile), elevation(s + step, profile)
        d0, d1 = slope(s, profile), slope(s + step, profile)
        c = 3 * (z1 - z0) / step**2 - (2 * d0 + d1) / step
        d = 2 * (z0 - z1) / step**3 + (d0 + d1) / step**2
        ET.SubElement(
            elevations,
            "elevation",
            s=f"{s:.6f}",
            a=f"{z0:.12g}",
            b=f"{d0:.12g}",
            c=f"{c:.12g}",
            d=f"{d:.12g}",
        )
    ET.SubElement(road, "lateralProfile")
    lanes = ET.SubElement(road, "lanes")
    section = ET.SubElement(lanes, "laneSection", s="0")
    center = ET.SubElement(section, "center")
    lane_zero = ET.SubElement(center, "lane", id="0", type="none", level="false")
    ET.SubElement(
        lane_zero,
        "roadMark",
        sOffset="0",
        type="solid",
        weight="standard",
        color="white",
        width="0.12",
    )
    right = ET.SubElement(section, "right")
    lane = ET.SubElement(right, "lane", id="-1", type="driving", level="false")
    ET.SubElement(lane, "link")
    ET.SubElement(lane, "width", sOffset="0", a="4", b="0", c="0", d="0")
    ET.SubElement(
        lane, "roadMark", sOffset="0", type="solid", weight="standard", color="white", width="0.12"
    )
    if parallel:
        second = copy.deepcopy(road)
        second.set("id", "2")
        second.set("name", "Identical passive comparison track")
        second.find("planView/geometry").set("y", "-8")
        root.append(second)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def generate_world(client, carla, profile, output, parallel=False):
    """Reset an otherwise empty world; persist the exact mesh input for replay."""
    if list(client.get_world().get_actors().filter("vehicle.*")):
        raise RuntimeError("Another vehicle simulation is running; cannot replace its world")
    text = opendrive(profile, parallel=parallel)
    (output / "road.xodr").write_text(text, encoding="utf-8")
    world = client.generate_opendrive_world(
        text,
        carla.OpendriveGenerationParameters(
            vertex_distance=0.05,
            max_road_length=20,
            wall_height=0.0,
            additional_width=0.0,
            smooth_junctions=False,
            enable_mesh_visibility=True,
        ),
    )
    return world


def verify_mesh(
    world, carla, profile, output, center_y=LANE_CENTER_Y_M, filename="road_mesh_check.json"
):
    """Independent vertical scene intersections against the authored profile."""
    rows = []
    for index in range(1, 601):
        station = index * 0.2
        hits = world.cast_ray(
            carla.Location(station, center_y, 2),
            carla.Location(station, center_y, -2),
        )
        if not hits:
            raise RuntimeError("Generated road has no collision surface at station " + str(station))
        expected = elevation(station, profile)
        actual = min((hit.location.z for hit in hits), key=lambda z: abs(z - expected))
        rows.append(
            dict(station_m=station, expected_m=expected, mesh_m=actual, error_m=actual - expected)
        )
    maximum = max(abs(row["error_m"]) for row in rows)
    if maximum > 0.0005:
        raise RuntimeError("Generated road height error exceeds 0.5 mm: " + str(maximum))
    import json

    report = dict(
        passed=True,
        center_y_m=float(center_y),
        samples=len(rows),
        max_height_error_m=maximum,
        min_height_m=min(row["mesh_m"] for row in rows),
        max_height_m=max(row["mesh_m"] for row in rows),
    )
    (output / filename).write_text(json.dumps(dict(report=report, samples=rows), indent=2))
    return report
