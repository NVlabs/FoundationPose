# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys,time,socket
code_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_path)
import open3d as o3d
import numpy as np
from PIL import Image
import cv2,imageio
import time
import trimesh
import pyrender
from transformations import *
import numpy as np
from PIL import Image
import cv2
import time
import argparse,pickle
from Utils import *


cvcam_in_glcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])

class ModelRendererOffscreen:
  def __init__(self, cam_K, H,W, zfar=100):
    '''
    @window_sizes: H,W
    '''
    self.K = cam_K
    self.scene = pyrender.Scene(ambient_light=[1., 1., 1.],bg_color=[0,0,0])
    self.camera = pyrender.IntrinsicsCamera(fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2],znear=0.1,zfar=zfar)
    self.cam_node = self.scene.add(self.camera, pose=np.eye(4), name='cam')
    self.mesh_nodes = []

    self.H = H
    self.W = W
    self.r = pyrender.OffscreenRenderer(self.W, self.H)


  def set_cam_pose(self, cam_pose):
    self.cam_node.matrix = cam_pose

  def add_mesh(self, mesh):
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    mesh_node = self.scene.add(mesh,pose=np.eye(4), name='ob') # Object pose parent is cam
    self.mesh_nodes.append(mesh_node)


  def add_point_light(self, intensity=3):
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    self.scene.add(light, pose=np.eye(4))  # Same as camera position


  def clear_mesh_nodes(self):
    for n in self.mesh_nodes:
      self.scene.remove_node(n)


  def render(self,mesh=None,ob_in_cvcam=None, get_normal=False):
    if mesh is not None:
      mesh = mesh.copy()
      mesh.apply_transform(cvcam_in_glcam@ob_in_cvcam)
      mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
      mesh_node = self.scene.add(mesh, pose=np.eye(4), name='ob') # Object pose parent is cam
    color, depth = self.r.render(self.scene)  # depth: float
    if mesh is not None:
      self.scene.remove_node(mesh_node)

    return color, depth
