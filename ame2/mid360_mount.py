"""Keep a proposed MID-360 URDF mount and its measurement transform consistent."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def measurement_position(position, rpy, optical_offset):
    r, p, y = rpy
    x, yy, z = optical_offset
    yy, z = math.cos(r)*yy - math.sin(r)*z, math.sin(r)*yy + math.cos(r)*z
    x, z = math.cos(p)*x + math.sin(p)*z, -math.sin(p)*x + math.cos(p)*z
    x, yy = math.cos(y)*x - math.sin(y)*yy, math.sin(y)*x + math.cos(y)*yy
    return [position[0]+x, position[1]+yy, position[2]+z]


def prepare_mount_urdf(source, output_dir, config):
    """Override only the requested fixed joint in a private copy of the asset."""
    proposal = config.get("urdf_mount_override")
    if proposal is None:
        return str(source)
    expected = measurement_position(proposal["position_m"], proposal["rpy_rad"],
                                    config["measurement_origin_in_mount_m"])
    if config["sensor_rpy_rad"] != proposal["rpy_rad"] or any(
            abs(a-b) > 1e-7 for a, b in zip(expected, config["sensor_translation_m"])):
        raise ValueError("Proposed URDF mount and LiDAR measurement transform disagree")
    source = Path(source).resolve()
    tree = ET.parse(source)
    joint = tree.getroot().find(f"joint[@name='{proposal['joint_name']}']")
    if joint is None or joint.get("type") != "fixed":
        raise ValueError("Expected the existing fixed MID-360 mount joint")
    origin = joint.find("origin")
    origin.set("xyz", " ".join(map(str, proposal["position_m"])))
    origin.set("rpy", " ".join(map(str, proposal["rpy_rad"])))
    for mesh in tree.getroot().iter("mesh"):
        name = mesh.get("filename")
        if "://" not in name:
            mesh.set("filename", str((source.parent / name).resolve()))
    output = Path(output_dir) / "proposed_mid360.urdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return str(output)
