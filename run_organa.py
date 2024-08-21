import os
import trimesh
import numpy as np
import logging
import argparse
from estimater import *
from datareader import *
import xml.etree.ElementTree as ET

class MeshObject:
    def __init__(self, mesh_name, mesh):
        self.mesh_name = mesh_name
        self.mesh = mesh
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)
        self.pose = None
        self.estimated = False

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
def calculate_class_average_precision(ious, class_labels, iou_threshold=0.3):
    """
    Calculate class-wise Average Precision (AP) for object detection or segmentation tasks using precomputed IoU values.

    Parameters:
    - ious: List of IoU values for matched predicted and ground truth boxes for all classes.
      Each element is a list of IoU values for a class.
    - class_labels: List of class labels for each IoU value (0, 1, 2, ...).
    - iou_threshold: Intersection over Union (IoU) threshold for matching predictions to ground truth.

    Returns:
    - Dictionary with class-wise AP values.
    - Mean Average Precision (mAP) across all classes.
    """

    class_ap = {}  # Store AP per class
    class_mAP = 0  # Initialize mean AP

    unique_classes = np.unique(class_labels)

    for class_idx in unique_classes:
        class_ious = [iou for iou, label in zip(ious, class_labels) if label == class_idx]
        
        num_predictions = len(class_ious)
        true_positives = np.zeros(num_predictions)
        false_positives = np.zeros(num_predictions)

        for i in range(num_predictions):
            if class_ious[i] >= iou_threshold:
                true_positives[i] = 1
            else:
                false_positives[i] = 1

        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_false_positives = np.cumsum(false_positives)

        precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
        print(precision)
        recall = cumulative_true_positives / num_predictions  # Using num_predictions for recall

        # Compute the average precision using the sum approximation
        class_ap[class_idx] = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        class_mAP += class_ap[class_idx]

    class_mAP /= len(unique_classes)  # Compute the mean AP (mAP)
    return class_ap
def read_bbox(f):
    tree = ET.parse(f)
    root = tree.getroot()
    filename = root.find("filename").text
    width = int(root.find(".//width").text)
    height = int(root.find(".//height").text)
    objects = []
    for obj in root.findall(".//object"):
        name = obj.find("name").text
        xmin = int(obj.find(".//xmin").text)
        ymin = int(obj.find(".//ymin").text)
        xmax = int(obj.find(".//xmax").text)
        ymax = int(obj.find(".//ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.2
IOU_THRESHOLD = 0.3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    mesh_files = ['conical_flask_250ml.obj', 'beaker_250ml.obj', 'conical_flask_500ml.obj', 'beaker_30ml.obj']
    meshes = [f'{code_dir}/perception_data/objects/{mesh}' for mesh in mesh_files]
    parser.add_argument('--mesh_files', type=str, nargs='+', default=meshes)
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/perception_data/black_paper')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    meshes = []
    for f in args.mesh_files:
        scene = trimesh.load(f)
        if isinstance(scene, trimesh.Scene):
            mesh = list(scene.geometry.values())[0]  # Access the first geometry object in the scene
        else:
            mesh = scene  # If it's already a Trimesh object
        meshes.append(mesh)
    mesh_names = [os.path.basename(f).replace('.obj', '') for f in args.mesh_files]
    debug = args.debug
    debug_dir = args.debug_dir
    mesh_objects = [MeshObject(mesh_name, mesh) for mesh_name, mesh in zip(mesh_names, meshes)]

    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    reader = OrganaReader(base_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # Initialize pose estimators for each object
    estimators = []
    
    for mesh_obj in mesh_objects:
        logging.info(f'Processing {mesh_obj.mesh_name}')
        
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(
            model_pts=mesh_obj.mesh.vertices, 
            model_normals=mesh_obj.mesh.vertex_normals, 
            mesh=mesh_obj.mesh, 
            scorer=scorer, 
            refiner=refiner, 
            debug_dir=debug_dir, 
            debug=debug, 
            glctx=glctx
        )
        estimators.append(est)
        logging.info(f"Estimator for {mesh_obj.mesh_name} initialized")

    i = 0
    
    while True:
        logging.info(f'Frame {i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        for mesh_obj in mesh_objects:
          reader.generate_mask(mesh_obj.mesh_name,i)
        if color is None or depth is None:
            logging.info("Waiting for color and depth images to be read...")
            continue
        
        for est, mesh_obj in zip(estimators, mesh_objects):
            if all([obj.estimated for obj in mesh_objects]):
                logging.info("All objects have been estimated, quitting...")
                break
            
            if not mesh_obj.estimated:
                if i == 0:
                    mask = reader.get_mask(i, mesh_obj.mesh_name, i).astype(bool)
                    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                    logging.info(f"Pose estimated for {mesh_obj.mesh_name}")
                    mesh_obj.pose = pose
                else:
                    pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
                    mesh_obj.pose = pose
                
                center_pose = mesh_obj.pose @ np.linalg.inv(mesh_obj.to_origin)
                if not is_object_in_frame(reader.K, center_pose, mesh_obj.bbox, color.shape):
                    mesh_obj.estimated = True
                    logging.info(f"Object {mesh_obj.mesh_name} is near the edge of the frame, discarding detection.")

                if debug >= 1:
                    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=mesh_obj.bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                    cv2.imshow('1', vis[..., ::-1])
                    cv2.waitKey(1)

                if debug >= 2:
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.jpg', vis)
        
        i += 1
