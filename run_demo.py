# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from tqdm.auto import tqdm

from pkm.util.torch_util import dcn

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  print('got mesh_file', args.mesh_file)
  # mesh = trimesh.load(args.mesh_file)
  mesh = trimesh.load(args.mesh_file,
                      force='mesh')

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  print('extents', extents)
  print('origin', to_origin)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in tqdm(range(len(reader.color_files))):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    print('color', color.shape)
    print('depth', depth.shape)
    if i==0:
      try:
          mask = reader.get_mask(0).astype(bool)
      except AttributeError:
          mask = None
      print('<register>')
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      print('</register>')

      if debug>=3:
        pose=pose.detach().cpu().numpy()
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.1
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      print('<track>')
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
      print('</track>')

    print('<save>')
    os.makedirs(f'{reader.video_dir}/ob_in_cam', exist_ok=True)
    pose = dcn(pose)
    np.savetxt(f'{reader.video_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    print('</save>')

    if debug>=1:
      print('<draw>')
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      if False:
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)
      else:
        os.makedirs(f'{reader.video_dir}/show', exist_ok=True)
        # np.savetxt(f'{reader.video_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
        cv2.imwrite(F'{reader.video_dir}/show/{reader.id_strs[i]}.png',vis[...,::-1])
      print('</draw>')


    if debug>=2:
      os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)

