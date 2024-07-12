#!/home/chemrobot/anaconda3/envs/foundationpose/bin/python
import os
import trimesh
import numpy as np
import logging
import argparse
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from estimater import *
from datareader import *
from pose_estimation_cfg import *

class Mesh():
    def __init__(self,mesh_name):
        self.mesh_name = mesh_name
        self.atedge = False
# Pose Publisher Function
def publish_pose_matrix(pose, mesh_name='cube'):
    pub = rospy.Publisher(f'pose_{mesh_name}', Float64MultiArray, queue_size=10)
    pose_flat = pose.flatten().tolist()
    
    pose_msg = Float64MultiArray()
    pose_msg.data = pose_flat
    
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
    
    pub.publish(pose_msg)

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

def is_object_in_frame(K, ob_in_cam, bbox, img_shape, margin=40):
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
    parser.add_argument('--mesh_files', type = str, default = meshes)
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/perception_data/test9')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    meshes = [trimesh.load(f) for f in args.mesh_files]
    mesh_names = [os.path.basename(f).replace('.obj', '') for f in args.mesh_files]
    debug = args.debug
    debug_dir = args.debug_dir
    mesh_objects = [Mesh(mesh_name) for mesh_name in mesh_names]

    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    rospy.init_node('pose_publisher', anonymous=True)
    
    # Initialize readers
    reader = OrganaReader(base_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    
    # Wait for the first set of messages
    color_image = rospy.wait_for_message('/zedm/zed_node/rgb/image_rect_color', Image)
    depth_image = rospy.wait_for_message('/zedm/zed_node/depth/depth_registered', Image)
    camera_info = rospy.wait_for_message('/zedm/zed_node/rgb/camera_info', CameraInfo)
    
    reader.color_callback(color_image)
    reader.depth_callback(depth_image)
    reader.camera_info_callback(camera_info)
    for mesh_name in mesh_names:
        rospy.loginfo(f"Mask generated for {mesh_name}")
        reader.generate_mask(mesh_name)
    
    print("Press Enter to continue...")
    while input() != '':
        pass
        

    # Start subscribers
    rospy.Subscriber('/zedm/zed_node/rgb/image_rect_color', Image, reader.color_callback)
    rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image, reader.depth_callback)
    rospy.Subscriber('/zedm/zed_node/rgb/camera_info', CameraInfo, reader.camera_info_callback)

    # Initialize pose estimators for each object
    estimators = []
    to_origins = []
    bboxes = []
    
    for mesh, mesh_name in zip(meshes, mesh_names):
        logging.info(f'Processing {mesh_name}')
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        to_origins.append(to_origin)
        bboxes.append(bbox)
        
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        estimators.append(est)
        logging.info(f"Estimator for {mesh_name} initialized")

    i = 0
    
    while not rospy.is_shutdown():
        logging.info(f'Frame {i}')
        color = reader.get_color()
        depth = reader.get_depth()
        mask = reader.get_mask(0, mesh_names[0])
        if color is None or depth is None or mask is None:
            rospy.loginfo("Waiting for color and depth images to be read...")
            continue
        
        for est, mesh_name, to_origin, bbox, mesh_object in zip(estimators, mesh_names, to_origins, bboxes, mesh_objects):
            # quit if all objects are at the edge
            if all([obj.atedge for obj in mesh_objects]):
                rospy.loginfo("All objects are at the edge, quitting...")
                quit()
            if not mesh_object.atedge:
                if i == 0:
                    mask = reader.get_mask(0, mesh_name).astype(bool)
                    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                    logging.info(f"Pose estimated for {mesh_name}")
                    publish_pose_matrix(pose, mesh_name=mesh_name)
                    logging.info(f"Pose published for {mesh_name}")
                    
                else:
                    pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
                    publish_pose_matrix(pose, mesh_name=mesh_name)
                
                center_pose = pose @ np.linalg.inv(to_origin)
                if not is_object_in_frame(reader.K, center_pose, bbox, color.shape):
                    mesh_object.atedge = True
                    rospy.loginfo(f"Object {mesh_name} is near the edge of the frame, discarding detection.")

                
                if debug >= 1:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                    cv2.imshow('1', vis[..., ::-1])
                    cv2.waitKey(1)

                if debug >= 2:
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[-1]}.jpg', vis)
        
        i += 1
