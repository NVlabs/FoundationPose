# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,uuid,joblib,os,sys,argparse
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml


def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
    valid = mask>0
  elif detect_type=='cnos':   #https://github.com/nv-nguyen/cnos
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cnos'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError

  return valid



def run_pose_estimation_worker(reader, i_frames, est:FoundationPose, debug=False, ob_id=None, device:int=0):
  result = NestDict()
  torch.cuda.set_device(device)
  est.to_device(f'cuda:{device}')
  est.glctx = dr.RasterizeCudaContext(device)
  debug_dir = est.debug_dir

  for i in range(len(i_frames)):
    i_frame = i_frames[i]
    id_str = reader.id_strs[i_frame]
    logging.info(f"{i}/{len(i_frames)}, video:{reader.get_video_id()}, id_str:{id_str}")
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)

    H,W = color.shape[:2]
    scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
    video_id = reader.get_video_id()

    logging.info(f"video:{reader.get_video_id()}, id_str:{id_str}, ob_id:{ob_id}")
    if ob_id not in scene_ob_ids:
      logging.info(f'skip {ob_id} as it does not exist in this scene')
      continue
    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id, iteration=5)
    logging.info(f"pose:\n{pose}")


    if debug>=3:
      tmp = est.mesh_ori.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result


def run_pose_estimation():
  wp.force_load(device='cuda')
  video_dirs = sorted(glob.glob(f'{opt.ycbv_dir}/test/*'))
  res = NestDict()

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir

  reader_tmp = YcbVideoReader(video_dirs[0])
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  ob_ids = reader_tmp.ob_ids

  for ob_id in ob_ids:
    if use_reconstructed_mesh:
      mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
    else:
      mesh = reader_tmp.get_gt_mesh(ob_id)
    symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]

    args = []
    for video_dir in video_dirs:
      reader = YcbVideoReader(video_dir, zfar=1.5)
      scene_ob_ids = reader.get_instance_ids_in_image(0)
      if ob_id not in scene_ob_ids:
        continue
      video_id = reader.get_video_id()

      for i in range(len(reader.color_files)):
        if not reader.is_keyframe(i):
          continue
        args.append((reader, [i], est, debug, ob_id, 0))

    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
    outs = []
    for arg in args:
      out = run_pose_estimation_worker(*arg)
      outs.append(out)

    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

  with open(f'{opt.debug_dir}/ycbv_res.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--ycbv_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video", help="data dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  opt = parser.parse_args()
  os.environ["YCB_VIDEO_DIR"] = opt.ycbv_dir

  set_seed(0)

  detect_type = 'mask'   # mask / box / detected

  run_pose_estimation()
