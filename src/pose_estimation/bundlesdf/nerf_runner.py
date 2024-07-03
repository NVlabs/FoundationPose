# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys,copy,cv2,itertools,uuid,joblib,uuid
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import imageio,trimesh
import json
import pdb
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(f'{code_dir}')
from nerf_helpers import *
from Utils import *


def batchify(fn, chunk):
  """Constructs a version of 'fn' that applies to smaller batches.
  """
  if chunk is None:
    return fn
  def ret(inputs):
    return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
  return ret


def compute_near_far_and_filter_rays(cam_in_world,rays,cfg):
  '''
  @cam_in_world: (4,4) in normalized space
  @rays: (...,D) in camera
  Return:
      (-1,D+2) with near far
  '''
  D = rays.shape[-1]
  rays = rays.reshape(-1,D)
  dirs_unit = rays[:,:3]/np.linalg.norm(rays[:,:3],axis=-1).reshape(-1,1)
  dirs = (cam_in_world[:3,:3]@rays[:,:3].T).T
  origins = (cam_in_world@to_homo(np.zeros(dirs.shape)).T).T[:,:3]
  bounds = np.array(cfg['bounding_box']).reshape(2,3)
  tmin,tmax = ray_box_intersection_batch(origins,dirs,bounds)
  tmin = tmin.data.cpu().numpy()
  tmax = tmax.data.cpu().numpy()
  ishit = tmin>=0
  near = (dirs_unit*tmin.reshape(-1,1))[:,2]
  far = (dirs_unit*tmax.reshape(-1,1))[:,2]
  good_rays = rays[ishit]
  near = near[ishit]
  far = far[ishit]
  near = np.abs(near)
  far = np.abs(far)
  good_rays = np.concatenate((good_rays,near.reshape(-1,1),far.reshape(-1,1)), axis=-1)  #(N,8+2)

  return good_rays

@torch.no_grad()
def sample_rays_uniform(N_samples,near,far,lindisp=False,perturb=True):
  '''
  @near: (N_ray,1)
  '''
  N_ray = near.shape[0]
  t_vals = torch.linspace(0., 1., steps=N_samples, device=near.device).reshape(1,-1)
  if not lindisp:
    z_vals = near * (1.-t_vals) + far * (t_vals)
  else:
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))   #(N_ray,N_sample)

  if perturb > 0.:
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=far.device)
    z_vals = lower + (upper - lower) * t_rand
    z_vals = torch.clip(z_vals,near,far)

  return z_vals.reshape(N_ray,N_samples)


class DataLoader:
  def __init__(self,rays,batch_size):
    self.rays = rays
    self.batch_size = batch_size
    self.pos = 0
    self.ids = torch.randperm(len(self.rays))

  def __next__(self):
    if self.pos+self.batch_size<len(self.ids):
      self.batch_ray_ids = self.ids[self.pos:self.pos+self.batch_size]
      out = self.rays[self.batch_ray_ids]
      self.pos += self.batch_size
      return out.cuda()

    self.ids = torch.randperm(len(self.rays))
    self.pos = self.batch_size
    self.batch_ray_ids = self.ids[:self.batch_size]
    return self.rays[self.batch_ray_ids].cuda()



class NerfRunner:
  def __init__(self,cfg,images,depths,masks,normal_maps,poses,K,_run=None,occ_masks=None,build_octree_pcd=None):
    set_seed(0)
    self.cfg = cfg
    self.cfg['tv_loss_weight'] = eval(str(self.cfg['tv_loss_weight']))
    self._run = _run
    self.images = images
    self.depths = depths
    self.masks = masks
    self.poses = poses
    self.normal_maps = normal_maps
    self.occ_masks = occ_masks
    self.K = K.copy()
    self.mesh = None
    self.train_pose = False
    self.N_iters = self.cfg['n_step']+1
    self.build_octree_pts = np.asarray(build_octree_pcd.points).copy()   # Make it pickable

    down_scale_ratio = cfg['down_scale_ratio']
    self.down_scale = np.ones((2),dtype=np.float32)
    if down_scale_ratio!=1:
      H,W = images[0].shape[:2]

      down_scale_ratio = int(down_scale_ratio)
      self.images = images[:, ::down_scale_ratio, ::down_scale_ratio]
      self.depths = depths[:, ::down_scale_ratio, ::down_scale_ratio]
      self.masks = masks[:, ::down_scale_ratio, ::down_scale_ratio]
      if normal_maps is not None:
        self.normal_maps = normal_maps[:, ::down_scale_ratio, ::down_scale_ratio]
      if occ_masks is not None:
        self.occ_masks = occ_masks[:, ::down_scale_ratio, ::down_scale_ratio]
      self.H, self.W = self.images.shape[1:3]
      self.cfg['dilate_mask_size'] = int(self.cfg['dilate_mask_size']//down_scale_ratio)

      self.K[0] *= float(self.W)/W
      self.K[1] *= float(self.H)/H
      self.down_scale = np.array([float(self.W)/W, float(self.H)/H])

    self.H, self.W = self.images[0].shape[:2]

    self.octree_m = None
    if self.cfg['use_octree']:
      self.build_octree()

    self.create_nerf()
    self.create_optimizer()

    self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])

    self.global_step = 0

    print("sc_factor",self.cfg['sc_factor'])
    print("translation",self.cfg['translation'])

    self.c2w_array = torch.tensor(poses).float().cuda()

    self.best_models = None
    self.best_loss = np.inf

    rays_ = []
    for i_mask in range(len(self.masks)):
      rays = self.make_frame_rays(i_mask)
      rays_.append(rays)
    rays = np.concatenate(rays_, axis=0)

    if self.cfg['denoise_depth_use_octree_cloud']:
      logging.info("denoise cloud")
      mask = (rays[:,self.ray_mask_slice]>0) & (rays[:,self.ray_depth_slice]<=self.cfg['far']*self.cfg['sc_factor'])
      rays_dir = rays[mask][:,self.ray_dir_slice]
      rays_depth = rays[mask][:,self.ray_depth_slice]
      pts3d = rays_dir*rays_depth.reshape(-1,1)
      frame_ids = rays[mask][:,self.ray_frame_id_slice].astype(int)
      pts3d_w = (self.poses[frame_ids]@to_homo(pts3d)[...,None])[:,:3,0]
      logging.info(f"Denoising rays based on octree cloud")

      kdtree = cKDTree(self.build_octree_pts)
      dists,indices = kdtree.query(pts3d_w,k=1,workers=-1)
      bad_mask = dists>0.02*self.cfg['sc_factor']
      bad_ids = np.arange(len(rays))[mask][bad_mask]
      rays[bad_ids,self.ray_depth_slice] = BAD_DEPTH*self.cfg['sc_factor']
      rays[bad_ids, self.ray_type_slice] = 1
      rays = rays[rays[:,self.ray_type_slice]==0]
      logging.info(f"bad_mask#={bad_mask.sum()}")

    rays = torch.tensor(rays, dtype=torch.float).cuda()

    self.rays = rays
    print("rays",rays.shape)
    self.data_loader = DataLoader(rays=self.rays, batch_size=self.cfg['N_rand'])


  def create_nerf(self,device=torch.device("cuda")):
    """Instantiate NeRF's MLP model.
    """
    models = {}
    embed_fn, input_ch = get_embedder(self.cfg['multires'], self.cfg, i=self.cfg['i_embed'], octree_m=self.octree_m)
    embed_fn = embed_fn.to(device)
    models['embed_fn'] = embed_fn

    input_ch_views = 0
    embeddirs_fn = None
    if self.cfg['use_viewdirs']:
      embeddirs_fn, input_ch_views = get_embedder(self.cfg['multires_views'], self.cfg, i=self.cfg['i_embed_views'], octree_m=self.octree_m)
    models['embeddirs_fn'] = embeddirs_fn

    output_ch = 4
    skips = [4]

    model = NeRFSmall(num_layers=2,hidden_dim=64,geo_feat_dim=15,num_layers_color=3,hidden_dim_color=64,input_ch=input_ch, input_ch_views=input_ch_views+self.cfg['frame_features']).to(device)
    model = model.to(device)
    models['model'] = model

    model_fine = None
    if self.cfg['N_importance'] > 0:
      if not self.cfg['share_coarse_fine']:
        model_fine = NeRFSmall(num_layers=2,hidden_dim=64,geo_feat_dim=15,num_layers_color=3,hidden_dim_color=64,input_ch=input_ch, input_ch_views=input_ch_views).to(device)
    models['model_fine'] = model_fine

    # Create feature array
    num_training_frames = len(self.images)
    feature_array = None
    if self.cfg['frame_features'] > 0:
      feature_array = FeatureArray(num_training_frames, self.cfg['frame_features']).to(device)
    models['feature_array'] = feature_array
    # Create pose array
    pose_array = None
    if self.cfg['optimize_poses']:
      pose_array = PoseArray(num_training_frames,max_trans=self.cfg['max_trans']*self.cfg['sc_factor'],max_rot=self.cfg['max_rot']).to(device)
    models['pose_array'] = pose_array
    self.models = models



  def make_frame_rays(self,frame_id):
    mask = self.masks[frame_id,...,0].copy()
    rays = get_camera_rays_np(self.H, self.W, self.K)   # [self.H, self.W, 3]  We create rays frame-by-frame to save memory
    rays = np.concatenate([rays, self.images[frame_id]], -1)  # [H, W, 6]
    rays = np.concatenate([rays, self.depths[frame_id]], -1)  # [H, W, 7]
    rays = np.concatenate([rays, self.masks[frame_id]>0], -1)  # [H, W, 8]
    if self.normal_maps is not None:
      rays = np.concatenate([rays, self.normal_maps[frame_id]], -1)  # [H, W, 11]
    rays = np.concatenate([rays, frame_id*np.ones(self.depths[frame_id].shape)], -1)  # [H, W, 12]
    ray_types = np.zeros((self.H,self.W,1))    # 0 is good; 1 is invalid depth (uncertain)
    invalid_depth = ((self.depths[frame_id,...,0]<self.cfg['near']*self.cfg['sc_factor']) | (self.depths[frame_id,...,0]>self.cfg['far']*self.cfg['sc_factor'])) & (mask>0)
    ray_types[invalid_depth] = 1
    rays = np.concatenate((rays,ray_types), axis=-1)
    self.ray_dir_slice = [0,1,2]
    self.ray_rgb_slice = [3,4,5]
    self.ray_depth_slice = 6
    self.ray_mask_slice = 7
    if self.normal_maps is not None:
      self.ray_normal_slice = [8,9,10]
      self.ray_frame_id_slice = 11
      self.ray_type_slice = 12
    else:
      self.ray_frame_id_slice = 8
      self.ray_type_slice = 9

    n = rays.shape[-1]

    ########## Option2: dilate
    down_scale_ratio = int(self.cfg['down_scale_ratio'])
    if frame_id==0:   #!NOTE first frame ob mask is assumed perfect
      kernel = np.ones((100, 100), np.uint8)
      mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
      if self.occ_masks is not None:
        mask[self.occ_masks[frame_id]>0] = 0
    else:
      dilate = 60//down_scale_ratio
      kernel = np.ones((dilate, dilate), np.uint8)
      mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
      if self.occ_masks is not None:
        mask[self.occ_masks[frame_id]>0] = 0


    if self.cfg['rays_valid_depth_only']:
      mask[invalid_depth] = 0

    vs,us = np.where(mask>0)
    cur_rays = rays[vs,us].reshape(-1,n)
    cur_rays = cur_rays[cur_rays[:,self.ray_type_slice]==0]
    cur_rays = compute_near_far_and_filter_rays(self.poses[frame_id],cur_rays,self.cfg)
    if self.normal_maps is not None:
      self.ray_near_slice = 13
      self.ray_far_slice = 14
    else:
      self.ray_near_slice = 10
      self.ray_far_slice = 11

    if self.cfg['use_octree']:
      rays_o_world = (self.poses[frame_id]@to_homo(np.zeros((len(cur_rays),3))).T).T[:,:3]
      rays_o_world = torch.from_numpy(rays_o_world).cuda().float()
      rays_unit_d_cam = cur_rays[:,:3]/np.linalg.norm(cur_rays[:,:3],axis=-1).reshape(-1,1)
      rays_d_world = (self.poses[frame_id][:3,:3]@rays_unit_d_cam.T).T
      rays_d_world = torch.from_numpy(rays_d_world).cuda().float()

      vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
      level = int(np.floor(np.log2(2.0/vox_size)))
      near,far,_,ray_depths_in_out = self.octree_m.ray_trace(rays_o_world,rays_d_world,level=level)
      near = near.cpu().numpy()
      valid = (near>0).reshape(-1)
      cur_rays = cur_rays[valid]

    return cur_rays


  def build_octree(self):
    if self.cfg['save_octree_clouds']:
      dir = f"{self.cfg['save_dir']}/build_octree_cloud.ply"
      pcd = toOpen3dCloud(self.build_octree_pts)
      o3d.io.write_point_cloud(dir,pcd)
      if self._run is not None:
        self._run.add_artifact(dir)
    pts = torch.tensor(self.build_octree_pts).cuda().float()                   # Must be within [-1,1]
    octree_smallest_voxel_size = self.cfg['octree_smallest_voxel_size']*self.cfg['sc_factor']
    finest_n_voxels = 2.0/octree_smallest_voxel_size
    max_level = int(np.round(np.log2(finest_n_voxels)))
    octree_smallest_voxel_size = 2.0/(2**max_level)

    #################### Dilate
    dilate_radius = int(np.round(self.cfg['octree_dilate_size']/octree_smallest_voxel_size))
    dilate_radius = max(1, dilate_radius)
    logging.info(f"Octree voxel dilate_radius:{dilate_radius}")
    shifts = []
    for dx in [-1,0,1]:
      for dy in [-1,0,1]:
        for dz in [-1,0,1]:
          shifts.append([dx,dy,dz])
    shifts = torch.tensor(shifts).cuda().long()    # (27,3)
    coords = torch.floor((pts+1)/octree_smallest_voxel_size).long()  #(N,3)
    dilated_coords = coords.detach().clone()
    for iter in range(dilate_radius):
      dilated_coords = (dilated_coords[None].expand(shifts.shape[0],-1,-1) + shifts[:,None]).reshape(-1,3)
      dilated_coords = torch.unique(dilated_coords,dim=0)
    pts = (dilated_coords+0.5) * octree_smallest_voxel_size - 1
    pts = torch.clip(pts,-1,1)

    if self.cfg['save_octree_clouds']:
      pcd = toOpen3dCloud(pts.data.cpu().numpy())
      dir = f"{self.cfg['save_dir']}/build_octree_cloud_dilated.ply"
      o3d.io.write_point_cloud(dir,pcd)
      if self._run is not None:
        self._run.add_artifact(dir)
    ####################

    assert pts.min()>=-1 and pts.max()<=1
    self.octree_m = OctreeManager(pts, max_level)

    if self.cfg['save_octree_clouds']:
      dir = f"{self.cfg['save_dir']}/octree_boxes_max_level.ply"
      m = self.octree_m.draw(level=max_level,method='point')
      m.export(dir)
      if self._run is not None:
        self._run.add_artifact(dir)
    vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
    level = int(np.round(np.log2(2.0/vox_size)))
    if self.cfg['save_octree_clouds']:
      dir = f"{self.cfg['save_dir']}/octree_boxes_ray_tracing_level.ply"
      m = self.octree_m.draw(level=level,method='point')
      m.export(dir)
      if self._run is not None:
        self._run.add_artifact(dir)


  def create_optimizer(self):
    params = []
    for k in self.models:
      if self.models[k] is not None and k!='pose_array':
          params += list(self.models[k].parameters())

    param_groups = [{'name':'basic', 'params':params, 'lr':self.cfg['lrate']}]
    if self.models['pose_array'] is not None:
      param_groups.append({'name':'pose_array', 'params':self.models['pose_array'].parameters(), 'lr':self.cfg['lrate_pose']})

    self.optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999),weight_decay=0,eps=1e-15)

    self.param_groups_init = copy.deepcopy(self.optimizer.param_groups)


  def save_weights(self,out_file,models):
    data = {
      'global_step': self.global_step,
      'model': models['model'].state_dict(),
      'optimizer': self.optimizer.state_dict(),
    }
    if 'model_fine' in models and models['model_fine'] is not None:
      data['model_fine'] = models['model_fine'].state_dict()
    if models['embed_fn'] is not None:
      data['embed_fn'] = models['embed_fn'].state_dict()
    if models['embeddirs_fn'] is not None:
      data['embeddirs_fn'] = models['embeddirs_fn'].state_dict()
    if self.cfg['optimize_poses']>0:
      data['pose_array'] = models['pose_array'].state_dict()
    if self.cfg['frame_features'] > 0:
      data['feature_array'] = models['feature_array'].state_dict()
    if self.octree_m is not None:
      data['octree'] = self.octree_m.octree
    dir = out_file
    torch.save(data,dir)
    print('Saved checkpoints at', dir)
    if self._run is not None:
      self._run.add_artifact(dir)
    dir1 = copy.deepcopy(dir)
    dir = f'{os.path.dirname(out_file)}/model_latest.pth'
    if dir1!=dir:
      os.system(f'cp {dir1} {dir}')
    if self._run is not None:
      self._run.add_artifact(dir)


  def schedule_lr(self):
    for i,param_group in enumerate(self.optimizer.param_groups):
      init_lr = self.param_groups_init[i]['lr']
      new_lrate = init_lr * (self.cfg['decay_rate'] ** (float(self.global_step) / self.N_iters))
      param_group['lr'] = new_lrate


  def render_images(self,img_i,cur_rays=None):
    if cur_rays is None:
      frame_ids = self.rays[:, self.ray_frame_id_slice].cuda()
      cur_rays = self.rays[frame_ids==img_i].cuda()
    gt_depth = cur_rays[:,self.ray_depth_slice]
    gt_rgb = cur_rays[:,self.ray_rgb_slice].cpu()
    ray_type = cur_rays[:,self.ray_type_slice].data.cpu().numpy()

    ori_chunk = self.cfg['chunk']
    self.cfg['chunk'] = copy.deepcopy(self.cfg['N_rand'])
    with torch.no_grad():
      rgb, extras = self.render(rays=cur_rays, lindisp=False,perturb=False,raw_noise_std=0, depth=gt_depth)
    self.cfg['chunk'] = ori_chunk

    sdf = extras['raw'][...,-1]
    z_vals = extras['z_vals']
    signs = sdf[:, 1:] * sdf[:, :-1]
    empty_rays = (signs>0).all(dim=-1)
    mask = signs<0
    inds = torch.argmax(mask.float(), axis=1)
    inds = inds[..., None]
    depth = torch.gather(z_vals,dim=1,index=inds)
    depth[empty_rays] = self.cfg['far']*self.cfg['sc_factor']
    depth = depth[..., None].data.cpu().numpy()

    rgb = rgb.data.cpu().numpy()

    rgb_full = np.zeros((self.H,self.W,3),dtype=float)
    depth_full = np.zeros((self.H,self.W),dtype=float)
    ray_mask_full = np.zeros((self.H,self.W,3),dtype=np.uint8)
    X = cur_rays[:,self.ray_dir_slice].data.cpu().numpy()
    X[:,[1,2]] = -X[:,[1,2]]
    projected = (self.K@X.T).T
    uvs = projected/projected[:,2].reshape(-1,1)
    uvs = uvs.round().astype(int)
    uvs_good = uvs[ray_type==0]
    ray_mask_full[uvs_good[:,1],uvs_good[:,0]] = [255,0,0]
    uvs_uncertain = uvs[ray_type==1]
    ray_mask_full[uvs_uncertain[:,1],uvs_uncertain[:,0]] = [0,255,0]
    rgb_full[uvs[:,1],uvs[:,0]] = rgb.reshape(-1,3)
    depth_full[uvs[:,1],uvs[:,0]] = depth.reshape(-1)
    gt_rgb_full = np.zeros((self.H,self.W,3),dtype=float)
    gt_rgb_full[uvs[:,1],uvs[:,0]] = gt_rgb.reshape(-1,3).data.cpu().numpy()
    gt_depth_full = np.zeros((self.H,self.W),dtype=float)
    gt_depth_full[uvs[:,1],uvs[:,0]] = gt_depth.reshape(-1).data.cpu().numpy()

    return rgb_full, depth_full, ray_mask_full, gt_rgb_full, gt_depth_full, extras


  def get_gradients(self):
    if self.models['pose_array'] is not None:
      max_pose_grad = torch.abs(self.models['pose_array'].data.grad).max()
    max_embed_grad = 0
    for embed in self.models['embed_fn'].embeddings:
      max_embed_grad = max(max_embed_grad,torch.abs(embed.weight.grad).max())
    if self.models['feature_array'] is not None:
      max_feature_grad = torch.abs(self.models['feature_array'].data.grad).max()
    return max_pose_grad, max_embed_grad, max_feature_grad


  def get_truncation(self):
    '''Annearl truncation over training
    '''
    if self.cfg['trunc_decay_type']=='linear':
      truncation = self.cfg['trunc_start'] - (self.cfg['trunc_start']-self.cfg['trunc']) * float(self.global_step)/self.cfg['n_step']
    elif self.cfg['trunc_decay_type']=='exp':
      lamb = np.log(self.cfg['trunc']/self.cfg['trunc_start']) / (self.cfg['n_step']/4)
      truncation = self.cfg['trunc_start']*np.exp(self.global_step*lamb)
      truncation = max(truncation,self.cfg['trunc'])
    else:
      truncation = self.cfg['trunc']

    truncation *= self.cfg['sc_factor']
    return truncation


  def train_loop(self,batch):
    target_s = batch[:, self.ray_rgb_slice]    # Color (N,3)
    target_d = batch[:, self.ray_depth_slice]    # Normalized scale (N)

    target_mask = batch[:,self.ray_mask_slice].bool().reshape(-1)
    frame_ids = batch[:,self.ray_frame_id_slice]

    rgb, extras = self.render(rays=batch, depth=target_d,lindisp=False,perturb=True,raw_noise_std=self.cfg['raw_noise_std'], get_normals=False)

    valid_samples = extras['valid_samples']   #(N_ray,N_samples)
    z_vals = extras['z_vals']  # [N_rand, N_samples + N_importance]
    sdf = extras['raw'][..., -1]

    N_rays,N_samples = sdf.shape[:2]
    valid_rays = (valid_samples>0).any(dim=-1).bool().reshape(N_rays) & (batch[:,self.ray_type_slice]==0)

    ray_type = batch[:,self.ray_type_slice].reshape(-1)
    ray_weights = torch.ones((N_rays), device=rgb.device, dtype=torch.float32)
    ray_weights[(frame_ids==0).view(-1)] = self.cfg['first_frame_weight']
    ray_weights = ray_weights*valid_rays.view(-1)
    sample_weights = ray_weights.view(N_rays,1).expand(-1,N_samples) * valid_samples
    img_loss = (((rgb-target_s)**2 * ray_weights.view(-1,1))).mean()
    rgb_loss = self.cfg['rgb_weight'] * img_loss
    loss = rgb_loss

    rgb0_loss = torch.tensor(0)
    if 'rgb0' in extras:
      img_loss0 = (((extras['rgb0']-target_s)**2 * ray_weights.view(-1,1))).mean()
      rgb0_loss = img_loss0*self.cfg['rgb_weight']
      loss += rgb0_loss

    depth_loss = torch.tensor(0)
    depth_loss0 = torch.tensor(0)
    if self.cfg['depth_weight']>0:
      signs = sdf[:, 1:] * sdf[:, :-1]
      mask = signs<0
      inds = torch.argmax(mask.float(), axis=1)
      inds = inds[..., None]
      z_min = torch.gather(z_vals,dim=1,index=inds)
      weights = ray_weights * (depth<=self.cfg['far']*self.cfg['sc_factor']) * (mask.any(dim=-1))
      depth_loss = ((z_min*weights-depth.view(-1,1)*weights)**2).mean() * self.cfg['depth_weight']
      loss = loss+depth_loss

    truncation = self.get_truncation()
    sample_weights[ray_type==1] = 0
    fs_loss, sdf_loss,empty_loss, front_mask,sdf_mask = get_sdf_loss(z_vals, target_d.reshape(-1,1).expand(-1,N_samples), sdf, truncation, self.cfg,return_mask=True, sample_weights=sample_weights, rays_d=batch[:,self.ray_dir_slice])
    fs_loss = fs_loss*self.cfg['fs_weight']
    empty_loss = empty_loss*self.cfg['empty_weight']
    sdf_loss = sdf_loss*self.cfg['trunc_weight']
    loss = loss + fs_loss + sdf_loss + empty_loss

    fs_rgb_loss = torch.tensor(0)
    if self.cfg['fs_rgb_weight']>0:
      fs_rgb_loss = ((((torch.sigmoid(extras['raw'][...,:3])-1)*front_mask[...,None])**2) * sample_weights[...,None]).mean()
      loss += fs_rgb_loss*self.cfg['fs_rgb_weight']

    eikonal_loss = torch.tensor(0)
    if self.cfg['eikonal_weight']>0:
      nerf_normals = extras['normals']
      eikonal_loss = ((torch.norm(nerf_normals[sdf<1], dim=-1)-1)**2).mean() * self.cfg['eikonal_weight']
      loss += eikonal_loss

    point_cloud_loss = torch.tensor(0)
    point_cloud_normal_loss = torch.tensor(0)


    reg_features = torch.tensor(0)
    if self.models['feature_array'] is not None:
      reg_features = self.cfg['feature_reg_weight'] * (self.models['feature_array'].data**2).mean()
      loss += reg_features

    if self.models['pose_array'] is not None:
      pose_array = self.models['pose_array']
      pose_reg = self.cfg['pose_reg_weight']*pose_array.data[1:].norm()
      loss += pose_reg

    variation_loss = torch.tensor(0)

    self.optimizer.zero_grad()
    self.amp_scaler.scale(loss).backward()

    self.amp_scaler.step(self.optimizer)
    self.amp_scaler.update()
    if self.global_step%10==0 and self.global_step>0:
      self.schedule_lr()

    if self.global_step%self.cfg['i_weights']==0 and self.global_step>0:
      self.save_weights(out_file=os.path.join(self.cfg['save_dir'], f'model_latest.pth'), models=self.models)

    if self.global_step % self.cfg['i_img'] == 0 and self.global_step>0:
      ids = torch.unique(self.rays[:, self.ray_frame_id_slice]).data.cpu().numpy().astype(int).tolist()
      ids.sort()
      last = ids[-1]
      ids = ids[::max(1,len(ids)//5)]
      if last not in ids:
        ids.append(last)
      canvas = []
      for frame_idx in ids:
        rgb, depth, ray_mask, gt_rgb, gt_depth, _ = self.render_images(frame_idx)
        mask_vis = (rgb*255*0.2 + ray_mask*0.8).astype(np.uint8)
        mask_vis = np.clip(mask_vis,0,255)
        rgb = np.concatenate((rgb,gt_rgb),axis=1)
        far = self.cfg['far']*self.cfg['sc_factor']
        gt_depth = np.clip(gt_depth, self.cfg['near']*self.cfg['sc_factor'], far)
        depth_vis = np.concatenate((to8b(depth / far), to8b(gt_depth / far)), axis=1)
        depth_vis = np.tile(depth_vis[...,None],(1,1,3))
        row = np.concatenate((to8b(rgb),depth_vis,mask_vis),axis=1)
        canvas.append(row)
      canvas = np.concatenate(canvas,axis=0).astype(np.uint8)
      dir = f"{self.cfg['save_dir']}/image_step_{self.global_step:07d}.png"
      imageio.imwrite(dir,canvas)
      if self._run is not None:
        self._run.add_artifact(dir)


    if self.global_step%self.cfg['i_print']==0:
      msg = f"Iter: {self.global_step}, valid_samples: {valid_samples.sum()}/{torch.numel(valid_samples)}, valid_rays: {valid_rays.sum()}/{torch.numel(valid_rays)}, "
      metrics = {
        'loss':loss.item(),
        'rgb_loss':rgb_loss.item(),
        'rgb0_loss':rgb0_loss.item(),
        'fs_rgb_loss': fs_rgb_loss.item(),
        'depth_loss':depth_loss.item(),
        'depth_loss0':depth_loss0.item(),
        'fs_loss':fs_loss.item(),
        'point_cloud_loss': point_cloud_loss.item(),
        'point_cloud_normal_loss':point_cloud_normal_loss.item(),
        'sdf_loss':sdf_loss.item(),
        'eikonal_loss': eikonal_loss.item(),
        "variation_loss": variation_loss.item(),
        'truncation(meter)': self.get_truncation()/self.cfg['sc_factor'],
        }
      if self.models['pose_array'] is not None:
        metrics['pose_reg'] = pose_reg.item()
      if 'feature_array' in self.models:
        metrics['reg_features'] = reg_features.item()
      for k in metrics.keys():
        msg += f"{k}: {metrics[k]:.7f}, "
      msg += "\n"
      logging.info(msg)

      if self._run is not None:
        for k in metrics.keys():
          self._run.log_scalar(k,metrics[k],self.global_step)

    if self.global_step % self.cfg['i_mesh'] == 0 and self.global_step > 0:
      with torch.no_grad():
        model = self.models['model_fine'] if self.models['model_fine'] is not None else self.models['model']
        mesh = self.extract_mesh(isolevel=0, voxel_size=self.cfg['mesh_resolution'])
        self.mesh = copy.deepcopy(mesh)
        if mesh is not None:
          dir = os.path.join(self.cfg['save_dir'], f'step_{self.global_step:07d}_mesh_normalized_space.obj')
          mesh.export(dir)
          if self._run is not None:
            self._run.add_artifact(dir)
          dir = os.path.join(self.cfg['save_dir'], f'step_{self.global_step:07d}_mesh_real_world.obj')
          if self.models['pose_array'] is not None:
            _,offset = get_optimized_poses_in_real_world(self.poses,self.models['pose_array'],translation=self.cfg['translation'],sc_factor=self.cfg['sc_factor'])
          else:
            offset = np.eye(4)
          mesh = mesh_to_real_world(mesh,offset,translation=self.cfg['translation'],sc_factor=self.cfg['sc_factor'])
          mesh.export(dir)
          if self._run is not None:
            self._run.add_artifact(dir)

    if self.global_step % self.cfg['i_pose'] == 0 and self.global_step > 0:
      if self.models['pose_array'] is not None:
        optimized_poses,offset = get_optimized_poses_in_real_world(self.poses,self.models['pose_array'],translation=self.cfg['translation'],sc_factor=self.cfg['sc_factor'])
      else:
        optimized_poses = self.poses
      dir = os.path.join(self.cfg['save_dir'], f'step_{self.global_step:07d}_optimized_poses.txt')
      np.savetxt(dir,optimized_poses.reshape(-1,4))
      if self._run is not None:
        self._run.add_artifact(dir)


  def train(self):
    set_seed(0)

    for iter in range(self.N_iters):
      if iter%(self.N_iters//10)==0:
        logging.info(f'train progress {iter}/{self.N_iters}')
      batch = next(self.data_loader)
      self.train_loop(batch.cuda())
      self.global_step += 1



  @torch.no_grad()
  def sample_rays_uniform_occupied_voxels(self,rays_d,depths_in_out,lindisp=False,perturb=False, depths=None, N_samples=None):
    '''We first connect the discontinuous boxes for each ray and treat it as uniform sample, then we disconnect into correct boxes
    @rays_d: (N_ray,3)
    @depths_in_out: Padded tensor each has (N_ray,N_intersect,2) tensor, the time travel of each ray
    '''
    N_rays = rays_d.shape[0]
    N_intersect = depths_in_out.shape[1]
    dirs = rays_d/rays_d.norm(dim=-1,keepdim=True)

    ########### Convert the time to Z
    z_in_out = depths_in_out.cuda()*torch.abs(dirs[...,2]).reshape(N_rays,1,1).cuda()

    if depths is not None:
      depths = depths.reshape(-1,1)
      trunc = self.get_truncation()
      valid = (depths>=self.cfg['near']*self.cfg['sc_factor']) & (depths<=self.cfg['far']*self.cfg['sc_factor']).expand(-1,N_intersect)
      valid = valid & (z_in_out>0).all(dim=-1)      #(N_ray, N_intersect)
      z_in_out[valid] = torch.clip(z_in_out[valid],
        min=torch.zeros_like(z_in_out[valid]),
        max=torch.ones_like(z_in_out[valid])*(depths.reshape(-1,1,1).expand(-1,N_intersect,2)[valid]+trunc))


    depths_lens = z_in_out[:,:,1]-z_in_out[:,:,0]   #(N_ray,N_intersect)
    z_vals_continous = sample_rays_uniform(N_samples,torch.zeros((N_rays,1),device=z_in_out.device).reshape(-1,1),depths_lens.sum(dim=-1).reshape(-1,1),lindisp=lindisp,perturb=perturb)     #(N_ray,N_sample)

    ############# Option2 mycuda extension
    N_samples = z_vals_continous.shape[1]
    z_vals = torch.zeros((N_rays,N_samples), dtype=torch.float, device=rays_d.device)
    z_vals = common.sampleRaysUniformOccupiedVoxels(z_in_out.contiguous(),z_vals_continous.contiguous(), z_vals)
    z_vals = z_vals.float().to(rays_d.device)    #(N_ray,N_sample)

    return z_vals,z_vals_continous


  def render_rays(self,ray_batch,retraw=True,lindisp=False,perturb=False,raw_noise_std=0.,depth=None, get_normals=False, tf=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction, frame_ids.
      model: function. Model for predicting RGB and density at each point
        in space.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to model_fine.
      model_fine: "fine" network with same spec as model.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
      @depth: depth (from depth image) values of the ray (N_ray,1)
      @tf: (N_ray,4,4) glcam in ob, normalized space
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_d = ray_batch[:,self.ray_dir_slice]
    rays_o = torch.zeros_like(rays_d)
    viewdirs = rays_d/rays_d.norm(dim=-1,keepdim=True)

    frame_ids = ray_batch[:,self.ray_frame_id_slice].long()

    tf = self.c2w_array[frame_ids]
    if self.models['pose_array'] is not None:
      tf = self.models['pose_array'].get_matrices(frame_ids)@tf

    rays_o_w = transform_pts(rays_o,tf)
    viewdirs_w = (tf[:,:3,:3]@viewdirs[:,None].permute(0,2,1))[:,:3,0]
    voxel_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
    level = int(np.floor(np.log2(2.0/voxel_size)))
    near,far,_,depths_in_out = self.octree_m.ray_trace(rays_o_w,viewdirs_w,level=level,debug=0)
    z_vals,_ = self.sample_rays_uniform_occupied_voxels(rays_d=viewdirs,depths_in_out=depths_in_out,lindisp=lindisp,perturb=perturb, depths=depth, N_samples=self.cfg['N_samples'])

    if self.cfg['N_samples_around_depth']>0 and depth is not None:      #!NOTE only fine when depths are all valid
      valid_depth_mask = (depth>=self.cfg['near']*self.cfg['sc_factor']) & (depth<=self.cfg['far']*self.cfg['sc_factor'])
      valid_depth_mask = valid_depth_mask.reshape(-1)
      trunc = self.get_truncation()
      near_depth = depth[valid_depth_mask]-trunc
      far_depth = depth[valid_depth_mask]+trunc*self.cfg['neg_trunc_ratio']
      z_vals_around_depth = torch.zeros((N_rays,self.cfg['N_samples_around_depth']), device=ray_batch.device).float()
      # if torch.sum(inside_mask)>0:
      z_vals_around_depth[valid_depth_mask] = sample_rays_uniform(self.cfg['N_samples_around_depth'],near_depth.reshape(-1,1),far_depth.reshape(-1,1),lindisp=lindisp,perturb=perturb)
      invalid_depth_mask = valid_depth_mask==0

      if invalid_depth_mask.any() and self.cfg['use_octree']:
        z_vals_invalid,_ = self.sample_rays_uniform_occupied_voxels(rays_d=viewdirs[invalid_depth_mask],depths_in_out=depths_in_out[invalid_depth_mask],lindisp=lindisp,perturb=perturb, depths=None, N_samples=self.cfg['N_samples_around_depth'])
        z_vals_around_depth[invalid_depth_mask] = z_vals_invalid
      else:
        z_vals_around_depth[invalid_depth_mask] = sample_rays_uniform(self.cfg['N_samples_around_depth'],near[invalid_depth_mask].reshape(-1,1),far[invalid_depth_mask].reshape(-1,1),lindisp=lindisp,perturb=perturb)

      z_vals = torch.cat((z_vals,z_vals_around_depth), dim=-1)
      valid_samples = torch.ones(z_vals.shape, dtype=torch.bool, device=ray_batch.device)   # During pose update if ray out of box, it becomes invalid

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    deformation = None
    raw,normals,valid_samples = self.run_network(pts, viewdirs, frame_ids, tf=tf, valid_samples=valid_samples, get_normals=get_normals)  # [N_rays, N_samples, 4]

    rgb_map, weights = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std=raw_noise_std, valid_samples=valid_samples, depth=depth)

    if self.cfg['N_importance'] > 0:
      rgb_map_0 = rgb_map

      for iter in range(self.cfg['N_importance_iter']):
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.cfg['N_importance'], det=(perturb==0.))
        z_samples = z_samples.detach()
        valid_samples_importance = torch.ones(z_samples.shape, dtype=torch.bool).to(z_vals.device)
        valid_samples_importance[torch.all(valid_samples==0, dim=-1).reshape(-1)] = 0

        if self.models['model_fine'] is not None and self.models['model_fine']!=self.models['model']:
          z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
          pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + self.cfg['N_importance'], 3]
          raw, normals,valid_samples = self.run_network(pts, viewdirs, frame_ids, tf=tf, valid_samples=valid_samples, get_normals=False)
        else:
          pts = rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :, None]  # [N_rays, N_samples + self.cfg['N_importance'], 3]
          raw_fine,valid_samples_importance = self.run_network(pts, viewdirs, frame_ids, tf=tf, valid_samples=valid_samples_importance, get_normals=False)
          z_vals = torch.cat([z_vals, z_samples], -1)  #(N_ray, N_sample)
          indices = torch.argsort(z_vals, dim=-1)
          z_vals = torch.gather(z_vals,dim=1,index=indices)
          raw = torch.gather(torch.cat([raw, raw_fine], dim=1), dim=1, index=indices[...,None].expand(-1,-1,raw.shape[-1]))
          valid_samples = torch.cat((valid_samples,valid_samples_importance), dim=-1)

        rgb_map, weights = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std=raw_noise_std,valid_samples=valid_samples)

    ret = {'rgb_map' : rgb_map, 'valid_samples':valid_samples, 'weights':weights, 'z_vals':z_vals}

    if retraw:
      ret['raw'] = raw

    if normals is not None:
      ret['normals'] = normals

    if deformation is not None:
      ret['deformation'] = deformation

    if self.cfg['N_importance'] > 0:
      ret['rgb0'] = rgb_map_0

    return ret


  def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, valid_samples=None, depth=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    truncation = self.get_truncation()
    if depth is not None:
      depth = depth.view(-1,1)

    if valid_samples is None:
      valid_samples = torch.ones(z_vals.shape, dtype=torch.bool).to(z_vals.device)

    def sdf2weights(sdf):
      sdf_from_depth = (depth.view(-1,1)-z_vals)/truncation
      weights = torch.sigmoid(sdf_from_depth*self.cfg['sdf_lambda']) * torch.sigmoid(-sdf_from_depth*self.cfg['sdf_lambda'])  # This not work well

      invalid = (depth>self.cfg['far']*self.cfg['sc_factor']).reshape(-1)
      mask = (z_vals-depth<=truncation*self.cfg['neg_trunc_ratio']) & (z_vals-depth>=-truncation)
      weights[~invalid] = weights[~invalid] * mask[~invalid]
      weights[invalid] = 0

      return weights / (weights.sum(dim=-1,keepdim=True) + 1e-10)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    weights = sdf2weights(raw[..., 3])

    weights[valid_samples==0] = 0
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    return rgb_map, weights


  def render(self, rays, depth=None,lindisp=False,perturb=False,raw_noise_std=0.0, get_normals=False):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [batch_size, 6]. Ray origin and direction for
        each example in batch.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates. Only true for llff data
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      @depth: depth values (N_ray,1)
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything.
    """
    all_ret = self.batchify_rays(rays,depth=depth,lindisp=lindisp,perturb=perturb,raw_noise_std=raw_noise_std, get_normals=get_normals)

    k_extract = ['rgb_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



  def batchify_rays(self,rays_flat, depth=None,lindisp=False,perturb=False,raw_noise_std=0.0, get_normals=False):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    chunk = self.cfg['chunk']
    for i in range(0, rays_flat.shape[0], chunk):
      if depth is not None:
        cur_depth = depth[i:i+chunk]
      else:
        cur_depth = None
      ret = self.render_rays(rays_flat[i:i+chunk],depth=cur_depth,lindisp=lindisp,perturb=perturb,raw_noise_std=raw_noise_std, get_normals=get_normals)
      for k in ret:
        if ret[k] is None:
          continue
        if k not in all_ret:
          all_ret[k] = []
        all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


  def run_network(self, inputs, viewdirs, frame_ids=None, tf=None, latent_code=None, valid_samples=None, get_normals=False):
    """Prepares inputs and applies network 'fn'.
    @inputs: (N_ray,N_sample,3) sampled points on rays in GL camera's frame
    @viewdirs: (N_ray,3) unit length vector in camera frame, z-axis backward
    @frame_ids: (N_ray)
    @tf: (N_ray,4,4)
    @latent_code: (N_ray, D)
    """
    N_ray,N_sample = inputs.shape[:2]

    if valid_samples is None:
      valid_samples = torch.ones((N_ray,N_sample), dtype=torch.bool, device=inputs.device)

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    tf_flat = tf[:,None].expand(-1,N_sample,-1,-1).reshape(-1,4,4)
    inputs_flat = transform_pts(inputs_flat, tf_flat)

    valid_samples = valid_samples.bool() & (torch.abs(inputs_flat)<=1).all(dim=-1).view(N_ray,N_sample).bool()

    embedded = torch.zeros((inputs_flat.shape[0],self.models['embed_fn'].out_dim), device=inputs_flat.device)
    if valid_samples is None:
      valid_samples = torch.ones((N_ray,N_sample), dtype=torch.bool, device=inputs_flat.device)

    if get_normals:
      if inputs_flat.requires_grad==False:
        inputs_flat.requires_grad = True

    with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
      if self.cfg['i_embed'] in [3]:
        embedded[valid_samples.reshape(-1)], valid_samples_embed = self.models['embed_fn'](inputs_flat[valid_samples.reshape(-1)])
        valid_samples = valid_samples.reshape(-1)
        prev_valid_ids = valid_samples.nonzero().reshape(-1)
        bad_ids = prev_valid_ids[valid_samples_embed==0]
        new_valid_ids = torch.ones((N_ray*N_sample),device=inputs.device).bool()
        new_valid_ids[bad_ids] = 0
        valid_samples = valid_samples & new_valid_ids
        valid_samples = valid_samples.reshape(N_ray,N_sample).bool()
      else:
        embedded[valid_samples.reshape(-1)] = self.models['embed_fn'](inputs_flat[valid_samples.reshape(-1)]).to(embedded.dtype)
    embedded = embedded.float()

    # Add latent code
    if self.models['feature_array'] is not None:
      if latent_code is None:
        frame_features = self.models['feature_array'](frame_ids)
        D = frame_features.shape[-1]
        frame_features = frame_features[:,None].expand(-1,N_sample,-1).reshape(-1,D)
      else:
        D = latent_code.shape[-1]
        frame_features = latent_code[:,None].expand(N_ray,N_sample,latent_code.shape[-1]).reshape(-1,D)
      embedded = torch.cat([embedded, frame_features], -1)

    # Add view directions
    if self.models['embeddirs_fn'] is not None:
      input_dirs = (tf[..., :3, :3]@viewdirs[...,None])[...,0]  #(N_ray,3)
      embedded_dirs = self.models['embeddirs_fn'](input_dirs)
      tmp = embedded_dirs.shape[1:]
      embedded_dirs_flat = embedded_dirs[:,None].expand(-1,N_sample,*tmp).reshape(-1,*tmp)
      embedded = torch.cat([embedded, embedded_dirs_flat], -1)

    outputs_flat = []
    with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
      chunk = self.cfg['netchunk']
      for i in range(0,embedded.shape[0],chunk):
        out = self.models['model'](embedded[i:i+chunk])
        outputs_flat.append(out)
    outputs_flat = torch.cat(outputs_flat, dim=0).float()
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]).float()

    normals = None
    if get_normals:
      sdf = outputs[...,-1]
      d_output = torch.zeros(sdf.shape, device=sdf.device)
      normals = torch.autograd.grad(outputs=sdf,inputs=inputs_flat,grad_outputs=d_output,create_graph=False,retain_graph=True,only_inputs=True,allow_unused=True)[0]
      normals = normals.reshape(N_ray,N_sample,3)

    return outputs,normals,valid_samples


  def run_network_density(self, inputs, get_normals=False):
    """Directly query the network w/o pose transformations or deformations (inputs are already in normalized [-1,1]); Particularly used for mesh extraction
    @inputs: (N,3) sampled points on rays in GL camera's frame
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    inputs_flat = torch.clip(inputs_flat,-1,1)
    valid_samples = torch.ones((len(inputs_flat)),device=inputs.device).bool()

    if not inputs_flat.requires_grad:
      inputs_flat.requires_grad = True

    with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
      if self.cfg['i_embed'] in [3]:
        embedded, valid_samples_embed = self.models['embed_fn'](inputs_flat)
        valid_samples = valid_samples.reshape(-1)
        prev_valid_ids = valid_samples.nonzero().reshape(-1)
        bad_ids = prev_valid_ids[valid_samples_embed==0]
        new_valid_ids = torch.ones((len(inputs_flat)),device=inputs.device).bool()
        new_valid_ids[bad_ids] = 0
        valid_samples = valid_samples & new_valid_ids
      else:
        embedded = self.models['embed_fn'](inputs_flat)
    embedded = embedded.float()
    input_ch = embedded.shape[-1]

    outputs_flat = []
    with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
      chunk = self.cfg['netchunk']
      for i in range(0,embedded.shape[0],chunk):
        alpha = self.models['model'].forward_sdf(embedded[i:i+chunk])   #(N,1)
        outputs_flat.append(alpha.reshape(-1,1))
    outputs_flat = torch.cat(outputs_flat,dim=0).float()
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    if get_normals:
      d_output = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
      normal = torch.autograd.grad(outputs=outputs,inputs=inputs_flat,grad_outputs=d_output,create_graph=False,retain_graph=True,only_inputs=True,allow_unused=True)[0]
      outputs = torch.cat((outputs, normal), dim=-1)

    return outputs,valid_samples


  @torch.no_grad()
  def extract_mesh(self, level=None, voxel_size=0.003, isolevel=0.0, return_sigma=False):
    voxel_size *= self.cfg['sc_factor']  # in "network space"

    bounds = np.array(self.cfg['bounding_box']).reshape(2,3)
    x_min, x_max = bounds[0,0], bounds[1,0]
    y_min, y_max = bounds[0,1], bounds[1,1]
    z_min, z_max = bounds[0,2], bounds[1,2]
    tx = np.arange(x_min+0.5*voxel_size, x_max, voxel_size)
    ty = np.arange(y_min+0.5*voxel_size, y_max, voxel_size)
    tz = np.arange(z_min+0.5*voxel_size, z_max, voxel_size)
    N = len(tx)
    query_pts = torch.tensor(np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), -1).astype(np.float32).reshape(-1,3)).float().cuda()

    if self.octree_m is not None:
      vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
      level = int(np.floor(np.log2(2.0/vox_size)))
      center_ids = self.octree_m.get_center_ids(query_pts, level)
      valid = center_ids>=0
    else:
      valid = torch.ones(len(query_pts), dtype=bool).cuda()

    logging.info(f'query_pts:{query_pts.shape}, valid:{valid.sum()}')
    flat = query_pts[valid]

    sigma = []
    chunk = self.cfg['netchunk']
    for i in range(0,flat.shape[0],chunk):
      inputs = flat[i:i+chunk]
      with torch.no_grad():
        outputs,valid_samples = self.run_network_density(inputs=inputs)
      sigma.append(outputs)
    sigma = torch.cat(sigma, dim=0)
    sigma_ = torch.ones((N**3)).float().cuda()
    sigma_[valid] = sigma.reshape(-1)
    sigma = sigma_.reshape(N,N,N).data.cpu().numpy()

    logging.info('Running Marching Cubes')
    from skimage import measure
    try:
      vertices, triangles, normals, values = measure.marching_cubes(sigma, isolevel)
    except Exception as e:
      logging.info(f"ERROR Marching Cubes {e}")
      return None

    logging.info(f'done V:{vertices.shape}, F:{triangles.shape}')

    voxel_size_ndc = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]]) / np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = voxel_size_ndc.reshape(1,3) * vertices[:, :3] + offset.reshape(1,3)

    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    if return_sigma:
      return mesh,sigma,query_pts

    return mesh


  def mesh_texture_from_train_images(self, mesh, rgbs_raw, tex_res=1024):
    '''
    @rgbs_raw: raw complete image that was trained on, no black holes
    @mesh: in normalized space
    '''
    assert len(self.images)==len(rgbs_raw)

    frame_ids = torch.arange(len(self.images)).long().cuda()
    tf = self.c2w_array[frame_ids]
    if self.models['pose_array'] is not None:
      tf = self.models['pose_array'].get_matrices(frame_ids)@tf
    tf = tf.data.cpu().numpy()
    from offscreen_renderer import ModelRendererOffscreen

    tex_image = torch.zeros((tex_res,tex_res,3)).cuda().float()
    weight_tex_image = torch.zeros(tex_image.shape[:-1]).cuda().float()
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh = mesh.unwrap()
    H,W = tex_image.shape[:2]
    uvs_tex = (mesh.visual.uv*np.array([W-1,H-1]).reshape(1,2))    #(n_V,2)

    renderer = ModelRendererOffscreen(cam_K=self.K, H=self.H, W=self.W, zfar=self.cfg['far']*self.cfg['sc_factor'])

    vertices_cuda = torch.from_numpy(mesh.vertices).float().cuda()
    faces_cuda = torch.from_numpy(mesh.faces).long().cuda()
    face_vertices = torch.zeros((len(faces_cuda),3,3))
    for i in range(3):
      face_vertices[:,i] = vertices_cuda[faces_cuda[:,i]]

    all_tri_list= {key: [] for key in range(mesh.triangles.shape[0])}
    for i in range(len(rgbs_raw)):
      cvcam_in_ob = tf[i]@np.linalg.inv(glcam_in_cvcam)
      _, render_depth = renderer.render(mesh=mesh, ob_in_cvcam=np.linalg.inv(cvcam_in_ob))
      xyz_map = depth2xyzmap(render_depth, self.K)
      mask = self.masks[i].reshape(self.H,self.W).astype(bool)
      valid = (render_depth.reshape(self.H,self.W)>=0.1*self.cfg['sc_factor']) & (mask)
      pts = xyz_map[valid].reshape(-1,3)
      pts = transform_pts(pts, cvcam_in_ob)
      ray_colors = rgbs_raw[i][valid].reshape(-1,3)
      locations, distance, index_tri = trimesh.proximity.closest_point(mesh, pts)
      normals = mesh.face_normals[index_tri]
      for ind_tri, each_tri in enumerate(index_tri):
          rays_o = np.zeros(3)
          pts = transform_pts(rays_o, cvcam_in_ob) #transform to world space
          rays_d = locations[ind_tri]-pts
          rays_d /= np.linalg.norm(rays_d)
          dot_product = np.dot(-rays_d, normals[ind_tri])
          angle_radians = np.arccos(dot_product)
          angle_degrees = np.degrees(angle_radians)
          all_tri_list[each_tri].extend([[i,angle_degrees]])

    _CHOOSE_TOP_N = 4
    all_triangles_dict={}
    for k,v in all_tri_list.items():
      if(v):
        v.sort(key=lambda x: x[1])
        tep = [i[0] for i in v[:_CHOOSE_TOP_N]]
        all_triangles_dict[k]=set(list(tep))


    all_tri_visited= {key: 0 for key in range(mesh.triangles.shape[0])}

    logging.info(f"Texture: Texture map computation")
    for i in range(len(rgbs_raw)):
      print(f'project train_images {i}/{len(rgbs_raw)}')

      cvcam_in_ob = tf[i]@np.linalg.inv(glcam_in_cvcam)
      _, render_depth = renderer.render(mesh=mesh, ob_in_cvcam=np.linalg.inv(cvcam_in_ob))
      xyz_map = depth2xyzmap(render_depth, self.K)
      mask = self.masks[i].reshape(self.H,self.W).astype(bool)
      valid = (render_depth.reshape(self.H,self.W)>=0.1*self.cfg['sc_factor']) & (mask)
      pts = xyz_map[valid].reshape(-1,3)
      pts = transform_pts(pts, cvcam_in_ob)
      ray_colors = rgbs_raw[i][valid].reshape(-1,3)
      locations, distance, index_tri = trimesh.proximity.closest_point(mesh, pts)
      normals = mesh.face_normals[index_tri]
      rays_o = np.zeros((len(normals),3))
      rays_o = transform_pts(rays_o,cvcam_in_ob)
      rays_d = locations-rays_o
      rays_d /= np.linalg.norm(rays_d,axis=-1).reshape(-1,1)
      dots = (normals*(-rays_d)).sum(axis=-1)
      ray_weights = np.ones((len(rays_o)))
      bool_weights=torch.zeros(len(locations), dtype=torch.bool, device='cuda')
      count = 0
      for jj, trtind__ in enumerate(index_tri):
        if(i in all_triangles_dict[trtind__] ):
            bool_weights[jj]=1
            all_tri_visited[trtind__]=1
            count +=1

      uvs = torch.zeros((len(locations),2)).cuda().float()
      common.rayColorToTextureImageCUDA(torch.from_numpy(mesh.faces).cuda().long(), torch.from_numpy(mesh.vertices).cuda().float(), torch.from_numpy(locations).cuda().float(), torch.from_numpy(index_tri).cuda().long(), torch.from_numpy(uvs_tex).cuda().float(), uvs)
      uvs = torch.round(uvs).long()
      uvs_flat = uvs[:,1]*(W-1) + uvs[:,0]
      uvs_flat_unique, inverse_ids, cnts = torch.unique(uvs_flat, return_counts=True, return_inverse=True)
      perm = torch.arange(inverse_ids.size(0)).cuda()
      inverse_ids, perm = inverse_ids.flip([0]), perm.flip([0])
      unique_ids = inverse_ids.new_empty(uvs_flat_unique.size(0)).scatter_(0, inverse_ids, perm)
      uvs_unique = torch.stack((uvs_flat_unique%(W-1), uvs_flat_unique//(W-1)), dim=-1).reshape(-1,2)
      cur_weights= bool_weights[unique_ids].cuda().float()
      tex_image[uvs_unique[:,1],uvs_unique[:,0]] += torch.from_numpy(ray_colors).cuda().float()[unique_ids]*cur_weights.reshape(-1,1)
      weight_tex_image[uvs_unique[:,1], uvs_unique[:,0]] += cur_weights

    tex_image = tex_image/weight_tex_image[...,None]
    tex_image = tex_image.data.cpu().numpy()
    tex_image = np.clip(tex_image,0,255).astype(np.uint8)
    tex_image = tex_image[::-1].copy()
    new_texture = texture_map_interpolation(tex_image)

    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv,image=Image.fromarray(new_texture))
    return mesh
