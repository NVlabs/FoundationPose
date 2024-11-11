# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import joblib,json,gzip,pickle
from sklearn.cluster import DBSCAN
import shutil,re,imageio,pdb,os,sys
from Utils import *
import pandas as pd


def find_biggest_cluster(pts, eps=0.06, min_samples=1):
  dbscan = DBSCAN(eps=eps,min_samples=min_samples,n_jobs=-1)
  dbscan.fit(pts)
  ids, cnts = np.unique(dbscan.labels_, return_counts=True)
  best_id = ids[cnts.argsort()[-1]]
  keep_mask = dbscan.labels_==best_id
  pts_cluster = pts[keep_mask]
  return pts_cluster, keep_mask


def compute_translation_scales(pts,max_dim=2,cluster=True, eps=0.06, min_samples=1):
  if cluster:
    pts, keep_mask = find_biggest_cluster(pts, eps, min_samples)
  else:
    keep_mask = np.ones((len(pts)), dtype=bool)
  max_xyz = pts.max(axis=0)
  min_xyz = pts.min(axis=0)
  center = (max_xyz+min_xyz)/2
  sc_factor = max_dim/(max_xyz-min_xyz).max()   #Normalize to [-1,1]
  sc_factor *= 0.9
  translation_cvcam = -center
  return translation_cvcam, sc_factor, keep_mask


def compute_scene_bounds_worker(color_file,K,glcam_in_world,use_mask,rgb=None,depth=None,mask=None):
  if rgb is None:
    depth_file = color_file.replace('images','depth_filtered')
    mask_file = color_file.replace('images','masks')
    rgb = np.array(Image.open(color_file))[...,:3]
    depth = cv2.imread(depth_file,-1)/1e3
  xyz_map = depth2xyzmap(depth,K)
  valid = depth>=0.1
  if use_mask:
    if mask is None:
      mask = cv2.imread(mask_file,-1)
    valid = valid & (mask>0)
  pts = xyz_map[valid].reshape(-1,3)
  if len(pts)==0:
    return None
  colors = rgb[valid].reshape(-1,3)
  pcd = toOpen3dCloud(pts,colors)
  pcd = pcd.voxel_down_sample(0.01)
  pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
  cam_in_world = glcam_in_world@glcam_in_cvcam
  pcd.transform(cam_in_world)
  return np.asarray(pcd.points).copy(), np.asarray(pcd.colors).copy()


def compute_scene_bounds(color_files,glcam_in_worlds,K,use_mask=True,base_dir=None,rgbs=None,depths=None,masks=None,cluster=True, translation_cvcam=None, sc_factor=None, eps=0.06, min_samples=1):
  assert color_files is None or rgbs is None

  if base_dir is None:
    base_dir = os.path.dirname(color_files[0])+'/../'

  args = []
  if rgbs is not None:
    for i in range(len(rgbs)):
      args.append((None,K,glcam_in_worlds[i],use_mask,rgbs[i],depths[i],masks[i]))
  else:
    for i in range(len(color_files)):
      args.append((color_files[i],K,glcam_in_worlds[i],use_mask))

  logging.info(f"compute_scene_bounds_worker start")
  ret = joblib.Parallel(n_jobs=10, prefer="threads")(joblib.delayed(compute_scene_bounds_worker)(*arg) for arg in args)
  logging.info(f"compute_scene_bounds_worker done")

  pcd_all = None
  for r in ret:
    if r is None:
      continue
    if pcd_all is None:
      pcd_all = toOpen3dCloud(r[0],r[1])
    else:
      pcd_all += toOpen3dCloud(r[0],r[1])
  pcd = pcd_all.voxel_down_sample(eps/5)

  logging.info(f"merge pcd")

  o3d.io.write_point_cloud(f'{base_dir}/naive_fusion.ply',pcd)
  pts = np.asarray(pcd.points).copy()

  def make_tf(translation_cvcam, sc_factor):
    tf = np.eye(4)
    tf[:3,3] = translation_cvcam
    tf1 = np.eye(4)
    tf1[:3,:3] *= sc_factor
    tf = tf1@tf
    return tf

  if translation_cvcam is None:
    translation_cvcam, sc_factor, keep_mask = compute_translation_scales(pts, cluster=cluster, eps=eps, min_samples=min_samples)
    tf = make_tf(translation_cvcam, sc_factor)
  else:
    tf = make_tf(translation_cvcam, sc_factor)
    tmp = copy.deepcopy(pcd)
    tmp.transform(tf)
    tmp_pts = np.asarray(tmp.points)
    keep_mask = (np.abs(tmp_pts)<1).all(axis=-1)

  logging.info(f"compute_translation_scales done")

  pcd = toOpen3dCloud(pts[keep_mask],np.asarray(pcd.colors)[keep_mask])
  o3d.io.write_point_cloud(f"{base_dir}/naive_fusion_biggest_cluster.ply",pcd)
  pcd_real_scale = copy.deepcopy(pcd)
  print(f'translation_cvcam={translation_cvcam}, sc_factor={sc_factor}')
  with open(f'{base_dir}/normalization.yml','w') as ff:
    tmp = {
      'translation_cvcam':translation_cvcam.tolist(),
      'sc_factor':float(sc_factor),
    }
    yaml.dump(tmp,ff)

  pcd.transform(tf)
  return sc_factor, translation_cvcam, pcd_real_scale, pcd


