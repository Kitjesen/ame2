"""Export actual live Thunder visual meshes and sensor returns for offline plots."""
import numpy as np
import torch
import warp as wp
from pxr import Usd, UsdGeom
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply, matrix_from_quat
from LidarSensor.example.isaaclab.isaaclab.sensors.isolated_geometry import raw_mesh
from ame2.lidar_mapping import transform_points, world_to_local


def snapshot(robot, sensor, body, points, accepted, tf, poses, truth, raw, observed, index):
    k = index
    meshes = []
    geometry = body._geometry
    positions, rotations = body._get_live_poses(geometry.view, None)
    for i, path in enumerate(geometry.mesh_paths):
        if f"/env_{k}/" not in path:
            continue
        mesh = geometry.meshes[i]
        local = wp.to_torch(mesh.points)
        anchor = geometry.anchor_indices[i]
        world = quat_apply(rotations[anchor].expand(len(local), -1), local) + positions[anchor]
        vertices = world_to_local(world[None], poses[k:k+1])[0]
        meshes.append({"name": path, "vertices": vertices.cpu(),
                       "faces": wp.to_torch(mesh.indices).reshape(-1, 3).cpu()})
    stage = get_current_stage()
    parent = stage.GetPrimAtPath(f"/World/envs/env_{k}/Robot/base_link")
    lidar = stage.GetPrimAtPath(str(parent.GetPath()) + "/visuals/lidar1_Link")
    cache = UsdGeom.XformCache()
    parent_inverse = cache.GetLocalToWorldTransform(parent).GetInverse()
    relative = np.asarray(cache.GetLocalToWorldTransform(lidar) * parent_inverse)
    expected = matrix_from_quat(torch.tensor(sensor.cfg.offset.rot)).numpy()
    np.testing.assert_allclose(relative[:3, :3].T, expected, atol=1e-5)
    for prim in Usd.PrimRange(lidar, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        vertices, faces = raw_mesh(prim)
        relative_mesh = np.asarray(cache.GetLocalToWorldTransform(prim) * parent_inverse)
        local = (np.column_stack((vertices, np.ones(len(vertices)))) @ relative_mesh)[:, :3]
        local = torch.as_tensor(local, device=poses.device, dtype=torch.float32)
        world = quat_apply(robot.data.root_quat_w[k].expand(len(local), -1), local) + poses[k, :3]
        vertices = world_to_local(world[None], poses[k:k+1])[0]
        meshes.append({"name": "lidar1_emitter", "vertices": vertices.cpu(),
                       "faces": torch.tensor(faces.reshape(-1, 3))})
    local_points = world_to_local(transform_points(points, tf), poses)[k]
    keep = accepted[k] & (local_points[:, 0] > -.5) & (local_points[:, 0] < 2.2) & (local_points[:, 1].abs() < 1.2)
    return {"sample_index": k, "meshes": meshes, "points": local_points[keep].cpu(),
            "truth": truth[k].cpu(), "raw": raw[k].cpu(), "observed": observed[k].cpu(),
            "mesh_rotation_matches_sensor": True}
