#!/home/chemrobot/anaconda3/envs/foundationpose/bin/python
import os
import trimesh
import numpy as np
import logging
import argparse
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from estimater import *
from datareader import *


# Pose Publisher Function
def publish_pose_matrix(pose, mesh_name='cube'):
    # publish the pose matrix as a flattened tensor
    pub = rospy.Publisher(f'pose_{mesh_name}', Float64MultiArray, queue_size=10)
    pose_flat = pose.flatten().tolist()
    
    # Create the MultiArray message
    pose_msg = Float64MultiArray()
    pose_msg.data = pose_flat
    
    # Define the layout
    dim1 = MultiArrayDimension()
    dim1.label = "rows"
    dim1.size = 4
    dim1.stride = 16  # 4x4 matrix
    dim2 = MultiArrayDimension()
    dim2.label = "columns"
    dim2.size = 4
    dim2.stride = 4

    pose_msg.layout.dim = [dim1, dim2]
    pose_msg.layout.data_offset = 0
    
    # Publish the message
    pub.publish(pose_msg)
    # Spin
    # rospy.spin()
import numpy as np

def to_homo(pts):
    return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)

def project_bbox_2d(K, ob_in_cam, bbox):
    min_xyz = bbox.min(axis=0)
    max_xyz = bbox.max(axis=0)

    corners = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]]
    ])

    projected_corners = (K @ (ob_in_cam @ to_homo(corners).T).T[:, :3].T).T
    uv = projected_corners[:, :2] / projected_corners[:, 2].reshape(-1, 1)

    return np.round(uv).astype(int)

def is_object_in_frame(K, ob_in_cam, bbox, img_shape, margin=20):
    projected_bbox = project_bbox_2d(K, ob_in_cam, bbox)
    
    x_coords = projected_bbox[:, 0]
    y_coords = projected_bbox[:, 1]
    
    if (x_coords.min() < margin or x_coords.max() > (img_shape[1] - margin) or
        y_coords.min() < margin or y_coords.max() > (img_shape[0] - margin)):
        return False
    return True

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
    mesh_name = os.path.basename(args.mesh_file).replace('.obj', '')
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
    
     # Read one image and generate mask
    color_image = rospy.wait_for_message('/zedm/zed_node/rgb/image_rect_color', Image)
    depth_image = rospy.wait_for_message('/zedm/zed_node/depth/depth_registered', Image)
    camera_info = rospy.wait_for_message('/zedm/zed_node/rgb/camera_info', CameraInfo)
    
    reader.color_callback(color_image)
    reader.depth_callback(depth_image)
    reader.camera_info_callback(camera_info)
    reader.generate_mask()


    # Start subscribing continuously
    rospy.Subscriber('/zedm/zed_node/rgb/image_rect_color', Image, reader.color_callback) # set the frequency of the subscriber to 1 Hz
    rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image, reader.depth_callback)
    rospy.Subscriber('/zedm/zed_node/rgb/camera_info', CameraInfo, reader.camera_info_callback)
    
    # rospy.spin()

    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    
    # for i in range(len(reader.color_files)):
    i = 0
    # Wait for the first image to be read
    
    while True:
        logging.info(f'i: {i}')
        color = reader.get_color()
        depth = reader.get_depth()
        if color is None or depth is None:
            rospy.loginfo("Waiting for color and depth images to be read...")
            continue
        
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            #print(np.unique(depth))
            
            # depth[(mask == 1) & (depth > 0.8)] = 0.78
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            # extrinsic_mat = np.load(f'{args.test_scene_dir}/gt_poses/ext.npy')
            
            # world_pose = extrinsic_mat.dot(pose)
            print("Pose estimated")
            print(pose)
            publish_pose_matrix(pose, mesh_name=mesh_name)
            print("Pose published")
            
            # if debug >= 3:
            #     m = mesh.copy()
            #     m.apply_transform(pose)
            #     m.export(f'{debug_dir}/model_tf.obj')
            #     xyz_map = depth2xyzmap(depth, reader.K)
            #     valid = depth >= 0.1
            #     pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            #     o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
            # if obj is not in the scene, terminate
           
            # print("Pose tracked")
            # print(pose)
            publish_pose_matrix(pose, mesh_name=mesh_name)
            # print("Pose published")
        center_pose = pose @ np.linalg.inv(to_origin)
        if not is_object_in_frame(reader.K, center_pose, bbox, color.shape):
            rospy.loginfo("Object is near the edge of the frame, discarding detection.")
            break
        
        os.makedirs(f'{debug_dir}/ob_in_cam_organa', exist_ok=True)
        i = i +1
        # np.savetxt(f'{debug_dir}/ob_in_cam_organa/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[-1]}.jpg', vis)

