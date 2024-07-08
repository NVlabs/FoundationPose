#!/home/chemrobot/anaconda3/envs/foundationpose/bin/python
import os
import trimesh
import numpy as np
import logging
import argparse
import rospy
from geometry_msgs.msg import Pose
from estimater import *
from datareader import *


# Pose Publisher Function
def publish_pose_matrix(pose):
    pub = rospy.Publisher('pose_matrix', Pose, queue_size=10)
    rospy.init_node('pose_publisher', anonymous=True)
    # Convert the 4x4 matrix to a Pose message
    pose_msg = Pose()
    pose_msg.position.x = pose[3]
    pose_msg.position.y = pose[7]
    pose_msg.position.z = pose[11]
    pose_msg.orientation.x = pose[0]
    pose_msg.orientation.y = pose[1]
    pose_msg.orientation.z = pose[2]
    pose_msg.orientation.w = pose[15]
    # Publish the message
    pub.publish(pose_msg)
    # Spin
    # rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/perception_data/objects/beaker_250ml.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/perception_data/test9')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # if test_scene_dir exists, use it, otherwise mkdir it

    if not os.path.exists(args.test_scene_dir):
        os.makedirs(args.test_scene_dir)
    reader = OrganaReader(base_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    rospy.init_node('pose_publisher', anonymous=True)
    
    color_sub = rospy.Subscriber('/zedm/zed_node/rgb/image_rect_color', Image, reader.color_callback)
    reader.generate_mask()
    depth_sub = rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image, reader.depth_callback)
    camera_info_sub = rospy.Subscriber('/zedm/zed_node/rgb/camera_info', CameraInfo, reader.camera_info_callback)
    rospy.spin()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    
    for i in range(len(reader.color_files)):
        logging.info(f'i: {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if True:
            mask = reader.get_mask(0).astype(bool)
            print("!!!!!")
            print(np.unique(depth))
            
            depth[(mask == 1) & (depth > 0.8)] = 0.78
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            extrinsic_mat = np.load(f'{args.test_scene_dir}/gt_poses/ext.npy')
            
            world_pose = extrinsic_mat.dot(pose)
            print(world_pose)
            world_pose = world_pose.flatten()
            publish_pose_matrix(world_pose)
            print("Pose published")
            
            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
        
        os.makedirs(f'{debug_dir}/ob_in_cam_organa', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam_organa/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
            