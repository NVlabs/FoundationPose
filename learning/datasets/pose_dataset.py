# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set, Union
import numpy as np
import torch
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../../')
from Utils import *


@dataclass
class PoseData:
    """
    rgb: (h, w, 3) uint8
    depth: (bsz, h, w) float32
    bbox: (4, ) int
    K: (3, 3) float32
    """
    rgb: np.ndarray = None
    bbox: np.ndarray = None
    K: np.ndarray = None
    depth: Optional[np.ndarray] = None
    object_data = None
    mesh_diameter: float = None
    rgbA: np.ndarray = None
    rgbB: np.ndarray = None
    depthA: np.ndarray = None
    depthB: np.ndarray = None
    maskA = None
    maskB = None
    poseA: np.ndarray = None   #(4,4)
    target: float = None

    def __init__(self, rgbA=None, rgbB=None, depthA=None, depthB=None, maskA=None, maskB=None, normalA=None, normalB=None, xyz_mapA=None, xyz_mapB=None, poseA=None, poseB=None, K=None, target=None, mesh_diameter=None, tf_to_crop=None, crop_mask=None, model_pts=None, label=None, model_scale=None):
      self.rgbA = rgbA      #(H,W,3) or (H,W*n_view,3) when multiview
      self.rgbB = rgbB
      self.depthA = depthA
      self.depthB = depthB
      self.poseA = poseA
      self.poseB = poseB
      self.maskA = maskA
      self.maskB = maskB
      self.crop_mask = crop_mask
      self.normalA = normalA
      self.normalB = normalB
      self.xyz_mapA = xyz_mapA
      self.xyz_mapB = xyz_mapB
      self.target = target
      self.K = K
      self.mesh_diameter = mesh_diameter
      self.tf_to_crop = tf_to_crop
      self.model_pts = model_pts
      self.label = label
      self.model_scale = model_scale


@dataclass
class BatchPoseData:
    """
    rgbs: (bsz, 3, h, w) torch tensor uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    K: (bsz, 3, 3) float32
    """

    rgbs: torch.Tensor = None
    object_datas = None
    bboxes: torch.Tensor = None
    K: torch.Tensor = None
    depths: Optional[torch.Tensor] = None
    rgbAs = None
    rgbBs = None
    depthAs = None
    depthBs = None
    normalAs = None
    normalBs = None
    poseA = None  #(B,4,4)
    poseB = None
    targets = None  # Score targets, torch tensor (B)

    def __init__(self, rgbAs=None, rgbBs=None, depthAs=None, depthBs=None, normalAs=None, normalBs=None, maskAs=None, maskBs=None, poseA=None, poseB=None, xyz_mapAs=None, xyz_mapBs=None, tf_to_crops=None, Ks=None, crop_masks=None, model_pts=None, mesh_diameters=None, labels=None):
        self.rgbAs = rgbAs
        self.rgbBs = rgbBs
        self.depthAs = depthAs
        self.depthBs = depthBs
        self.normalAs = normalAs
        self.normalBs = normalBs
        self.poseA = poseA
        self.poseB = poseB
        self.maskAs = maskAs
        self.maskBs = maskBs
        self.xyz_mapAs = xyz_mapAs
        self.xyz_mapBs = xyz_mapBs
        self.tf_to_crops = tf_to_crops
        self.crop_masks = crop_masks
        self.Ks = Ks
        self.model_pts = model_pts
        self.mesh_diameters = mesh_diameters
        self.labels = labels


    def pin_memory(self) -> "BatchPoseData":
        for k in self.__dict__:
            if self.__dict__[k] is not None:
              try:
                self.__dict__[k] = self.__dict__[k].pin_memory()
              except Exception as e:
                pass
        return self

    def cuda(self):
        for k in self.__dict__:
            if self.__dict__[k] is not None:
              try:
                self.__dict__[k] = self.__dict__[k].cuda()
              except:
                pass
        return self

    def select_by_indices(self, ids):
      out = BatchPoseData()
      for k in self.__dict__:
        if self.__dict__[k] is not None:
          out.__dict__[k] = self.__dict__[k][ids.to(self.__dict__[k].device)]
      return out

