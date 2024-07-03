# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple,Union
import numpy as np
import omegaconf
import torch


@dataclass
class TrainingConfig(omegaconf.dictconfig.DictConfig):
    input_resize: tuple = (160, 160)
    normalize_xyz:Optional[bool] = True
    use_mask:Optional[bool] = False
    crop_ratio:Optional[float] = None
    split_objects_across_gpus: bool = True
    max_num_key: Optional[int] = None
    use_normal:bool = False
    n_view:int = 1
    zfar:float = np.inf
    c_in:int = 6
    train_num_pair:Optional[int] = None
    make_pair_online:Optional[bool] = False
    render_backend:Optional[str] = 'nvdiffrast'

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    resume_run_id: Optional[str] = None
    save_dir: Optional[str] = None
    batch_size: int = 64
    epoch_size: int = 115200
    val_size: int = 1280
    n_epochs: int = 25
    save_epoch_interval: int = 100
    n_dataloader_workers: int = 20
    n_rendering_workers: int = 1
    gradient_max_norm:float = np.inf
    max_step_per_epoch: Optional[int] = 25000

    # Network
    use_BN:bool = True
    loss_type:Optional[str] = 'pairwise_valid'

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1   # -1 means disable
    n_epochs_warmup: int = 1

    # Visualization
    vis_interval: Optional[int] = 1000

    debug: Optional[bool] = None



@dataclass
class TrainRefinerConfig:
    # Datasets
    input_resize: tuple = (160, 160)  #(W,H)
    crop_ratio:Optional[float] = None
    max_num_key: Optional[int] = None
    use_normal:bool = False
    use_mask:Optional[bool] = False
    normal_uint8:bool = False
    normalize_xyz:Optional[bool] = True
    trans_normalizer:Optional[list] = None
    rot_normalizer:Optional[float] = None
    c_in:int = 6
    n_view:int = 1
    zfar:float = np.inf
    trans_rep:str = 'tracknet'  # tracknet/deepim
    rot_rep:Optional[str] = 'axis_angle'  # 6d/axis_angle
    save_dir: Optional[str] = None

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    batch_size: int = 64
    use_BN:bool = True
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1
    loss_type:str = 'l2'   # l1/l2/add

    vis_interval: Optional[int] = 1000
    debug: Optional[bool] = None


