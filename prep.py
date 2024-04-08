#!/usr/bin/env python3

# rgb
# depth
# masks
# cam_K.txt

from pathlib import Path
import pickle
import cv2
import numpy as np
from yourdfpy import URDF
from pkm.util.path import ensure_directory
from pkm.real.cseg.fit import SegmenterApp
from pkm.util.path import get_path

out_dir = ensure_directory(
    '/tmp/docker/calib-fp'
)
with open('/tmp/docker/calib.pkl', 'rb') as fp:
    d = pickle.load(fp)
    print('d', d.keys())

if True:
    # == color ==
    for i in range(len(d['color'])):
        j = 0
        out_file = out_dir / F'{i}' / 'rgb' / F'{j:07d}.png'
        ensure_directory(Path(out_file).parent)
        cv2.imwrite(str(out_file), d['color'][i][..., ::-1])

    # == depth ==
    for i in range(len(d['depth'])):
        j = 0
        out_file = out_dir / F'{i}' / 'depth' / F'{j:07d}.png'
        ensure_directory(Path(out_file).parent)
        cv2.imwrite(str(out_file), (d['depth'][i] * 1000.0).astype(
            np.uint16))

    if False:
        # == label ==
        for i in range(len(d['color'])):
            j = 0
            out_file = out_dir / F'{i}' / 'masks' / F'{j:07d}.png'
            ensure_directory(Path(out_file).parent)
            SegmenterApp.run(SegmenterApp.Config(
                ckpt_file='/tmp/docker/repvit_sam.pt',
                sam_dir='/home/user/Grounded-Segment-Anything/'

            ),
                d['color'][i][..., ::-1], str(out_file))

    # == intrinsics ==
    for i in range(len(d['intrinsics'])):
        out_file = out_dir / F'{i}' / 'cam_K.txt'
        ensure_directory(Path(out_file).parent)
        np.savetxt(out_file, d['intrinsics'][i], delimiter=' ')

# == mesh ==
urdf_path = get_path(
    'assets/franka_description/robots/franka_panda_custom_v3.urdf')
urdf = URDF.load(urdf_path)
urdf.update_cfg(d['joint_pos'].mean(axis=0))
# print(urdf.scene.graph.nodes)
# print(urdf.scene.subscene('panda_link5'))
T = urdf.get_transform('panda_link5')
urdf.scene.dump(concatenate=True).export(F'{out_dir}/robot_full.obj')
urdf.scene.subscene('panda_link5').apply_transform(T).dump(
    concatenate=True).export(F'{out_dir}/robot.obj')
