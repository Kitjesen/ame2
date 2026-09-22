"""Compare Thunder's sensor mesh axes with the manufacturer's optical-frame CAD.

This estimates CAD alignment, not physical calibration. The optical-frame NPZ
is generated from Livox's STEP by the existing Go2 model preparation tool.
Requires the existing research runtime's NumPy/SciPy, not Isaac or a GPU.
"""
import argparse
import itertools
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thunder-stl", required=True)
    parser.add_argument("--official-npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = Path(args.thunder_stl).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    triangles = np.frombuffer(raw, dtype=[("normal","<f4",3),("v","<f4",(3,3)),("attr","<u2")], offset=84, count=count)
    source = np.unique(triangles["v"].reshape(-1,3), axis=0).astype(float)*1000
    target = np.load(args.official_npz)["vertices"]
    # Thunder omits the connector overhang; compare the common 65 mm body.
    target = target[(np.abs(target[:,:2]) <= 32.51).all(-1)]
    tree = cKDTree(target)
    results = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1), repeat=3):
            rotation = np.eye(3)[list(perm)] * np.array(signs)[:,None]
            if np.linalg.det(rotation) < 0: continue
            translation = (target.min(0)+target.max(0))/2 - rotation @ ((source.min(0)+source.max(0))/2)
            for _ in range(30):
                distance, indices = tree.query(source @ rotation.T + translation)
                good = distance <= np.quantile(distance, .85)
                a, b = source[good], target[indices[good]]
                u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
                fix = np.eye(3)
                fix[2,2] = np.linalg.det(vt.T @ u.T)
                rotation = vt.T @ fix @ u.T
                translation = b.mean(0) - rotation @ a.mean(0)
            residual = tree.query(source @ rotation.T + translation)[0]
            results.append({"trimmed_rmse_mm":float(np.mean(np.sort(residual)[:int(len(residual)*.85)]**2)**.5),
                            "rotation_link_to_optical":rotation.tolist(),
                            "translation_link_to_optical_mm":translation.tolist(),
                            "optical_z_in_link":rotation.T[:,2].tolist()})
    results.sort(key=lambda x:x["trimmed_rmse_mm"])
    record = {"scope":"CAD geometry only; no field calibration", "trimmed_fraction":.85,
              "official_display_quantization_mm":.25, "best_candidates":results[:5]}
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
