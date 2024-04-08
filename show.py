#!/usr/bin/env python3

import pickle
import numpy as np
import open3d as o3d
import trimesh
import cv2
from cho_util.math import transform as tx

with open('/tmp/T_cb.pkl', 'rb') as fp:
    T_cbs = pickle.load(fp)
pcdss = []
for i in range(4):
    # T_cb = np.loadtxt('/tmp/docker/calib-fp/0/ob_in_cam/0000000.txt')
    T_cb = T_cbs[i]
    color = cv2.imread(F'/tmp/docker/calib-fp/{i}/rgb/0000000.png')
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(F'/tmp/docker/calib-fp/{i}/depth/0000000.png',
                       cv2.IMREAD_UNCHANGED) / 1000.0
    K = np.loadtxt(F'/tmp/docker/calib-fp/{i}/cam_K.txt').astype(np.float32)

    # T_bc = tx.invert(T_cb).astype(np.float32)
    T_cb = T_cb.astype(np.float32)
    # print(T_bc)
    pcds = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(color.astype(np.uint8)),
            o3d.t.geometry.Image(depth.astype(np.float32))),
        K, depth_scale=1.0, with_normals=True, extrinsics=T_cb)

    pcdss.append(pcds)

robot = o3d.t.io.read_triangle_mesh(
    '/tmp/docker/calib-fp/robot_full.obj')
robot_ls = o3d.t.geometry.LineSet.from_legacy(
    o3d.geometry.LineSet.create_from_triangle_mesh(robot.to_legacy())
)
o3d.visualization.draw([*pcdss, robot, robot_ls])
