"""Prevent ray-only mount changes and protect the shared robot asset."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ame2.mid360_mount import measurement_position, prepare_mount_urdf


@pytest.mark.parametrize("name", ["ground", "side"])
def test_proposed_sensor_pose_matches_urdf_and_source_stays_unchanged(tmp_path, name):
    cfg = json.loads((Path(__file__).parents[1] / f"configs/thunder_v4_mid360_{name}_view.json").read_text())
    source = tmp_path / "source.urdf"
    contents = '<robot name="test"><link name="base"><visual><geometry><mesh filename="body.stl"/></geometry></visual></link><joint name="lidar1_joint" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/></joint></robot>'
    source.write_text(contents)
    output = prepare_mount_urdf(source, tmp_path / "private", cfg)
    assert source.read_text() == contents
    root = ET.parse(output).getroot()
    assert root.find("joint/origin").get("rpy") == " ".join(map(str, cfg["sensor_rpy_rad"]))
    assert Path(root.find("link/visual/geometry/mesh").get("filename")).is_absolute()
    expected = measurement_position(cfg["urdf_mount_override"]["position_m"], cfg["sensor_rpy_rad"],
                                    cfg["measurement_origin_in_mount_m"])
    assert expected == pytest.approx(cfg["sensor_translation_m"])
    cfg["sensor_translation_m"][2] += .01
    with pytest.raises(ValueError, match="disagree"):
        prepare_mount_urdf(source, tmp_path / "private", cfg)
