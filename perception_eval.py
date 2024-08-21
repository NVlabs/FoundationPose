#!/usr/bin/env python3
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_cluster

import glob
import cv2
import apriltag
import numpy as np
import pandas as pd
import torch
import json
import xml.etree.ElementTree as ET

from groundingdino.util.inference import load_model, predict, annotate, preprocess_image, annotate_nms

import matplotlib.pyplot as plt
import argparse
import time
import sys
import os
from models.cluster.util import non_maximum_suppression, iou_bbox
from models.cluster.cluster_model import draw_axes
from autolab_core import RigidTransform as rt
import open3d


model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "after_wait.jpg"
#TEXT_PROMPT = "wire . glass object . plate . scale"
#TEXT_PROMPT = "glass object . plate"
TEXT_PROMPT = "red apple . bowl . lemon"
#OBJECTIVE_PROMPTS = ['glass object', 'plate', 'glass']
OBJECTIVE_PROMPTS = ['bowl', 'lemon', 'red apple']
#OBJECTIVE_PROMPTS = ['glass object']
PROMPT_TO_TYPE_MAP = {
    'glass': ['glass object','glass'],
    'plate': ['plate']}


#BOX_TRESHOLD = 0.08
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.2
IOU_THRESHOLD = 0.3

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



def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_prompt_path', type=str, default='./data/')
    parser.add_argument('--img_prompt_cfg_path', type=str, default='./config/cluster.yaml')
    return parser.parse_args()

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

def map_prompt_to_object(label):
    for obj_type in PROMPT_TO_TYPE_MAP.keys():
        if label in PROMPT_TO_TYPE_MAP[obj_type]:
            return obj_type
    return 'Unknown'

def visualize_scene_pc(input_image, depth_map, cam_int):
    color_raw = open3d.geometry.Image(input_image)
    depth_raw =   open3d.geometry.Image((depth_map))
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    intrinsic = open3d.camera.PinholeCameraIntrinsic(width=640, height=480,
                                                    fx=cam_int[0][0],
                                                    fy=cam_int[1][1],
                                                    cx=cam_int[0][2],
                                                    cy=cam_int[1][2])

    #pcd = open3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    viewer = open3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = False
    #opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()

class PerceptionEval:
    def __init__(self, text_prompts, data_dir = 'perception_data'):
        self.metadata = pd.read_csv('metadata.csv')

        self.image_paths = glob.glob('perception_data/**/*.jpg', recursive=True)
        self.depth_paths = glob.glob('perception_data/**/depth_*.npy', recursive=True)
        self.camera_pose_paths = glob.glob('perception_data/**/camera_pose_*.npy', recursive=True)
        self.metadata_paths = glob.glob('perception_data/**/*.json', recursive=True)
        self.labels = glob.glob('perception_data/**/*.xml', recursive=True)
        self.gt_position_files = glob.glob('perception_data/**/*.tf', recursive=True)
        
        self.objects = pd.read_csv('objects.csv')
        self.args = init_args()
        self.camera_matrix = np.array([[729.42260742,   0.        , 617.55908203],
                                [  0.        , 729.42260742, 359.8135376 ],
                                [  0.        ,   0.        ,   1.        ]])
        self.knowledge_base = {}
        self.data_dir = data_dir
        self.text_prompts = text_prompts

        img_prompt_cfg = parse_config_utils.Config(config_path=self.args.img_prompt_cfg_path)
        self.cluster = build_cluster(cfg=img_prompt_cfg)
        self.occluded_ids = [
            1698287651,
            1698287496,
            1698287390,
            1698287525,
            1698441059,
            1698442292,
            1698442659,
            1698441882
        ]
        #self.selected_ids=[1698292730]
        #self.selected_ids=[1698292165]
        self.selected_ids=[1698292165]

    def evaluate(self):
        history = []
        idx = 0
        plate_ious = []
        object_2 = []
        object_5 = []
        object_14 = []
        object_17 = []
        object_23 = []

        for index, row in self.metadata.iterrows(): 
            #if row['data_id']  not in self.occluded_ids:  # whole dataset evaluation
            if row['data_id'] in self.selected_ids:
                idx+=1
                # if idx >8:
                #     break
                ground_truth_positions = [(path.split('/')[-1][:-3],rt().load(path)) for path in glob.glob(f'{row["root"]}/*.tf', recursive=True)]
                og_image = cv2.imread(f'{row["root"]}/image_{row["data_id"]}.jpg')
                og_image=cv2.imread("test.jpg")
                # plt.imshow(og_image)
                # plt.show()

                # depth_image = np.load(f'{row["root"]}/depth_{row["data_id"]}.npy')
                # plt.axis('off')
                # plt.imsave('raw_depth.jpg', depth_image)
                # plt.show()

                print(f'{row["root"]}/image_{row["data_id"]}.jpg')
                # plt.imshow(og_image)
                # plt.show()
                camera_pose = np.load(f'{row["root"]}/camera_pose_{row["data_id"]}.npy')
                depth = np.load(f'{row["root"]}/depth_{row["data_id"]}.npy')
                labels = read_bbox(f'{row["root"]}/image_{row["data_id"]}.xml')
                image = preprocess_image(og_image.copy())

                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )
                print(phrases)

                annotated_frame, xyxy = annotate(image_source=og_image, boxes=boxes, logits=logits, phrases=phrases)
                #plt.imsave('before_nms.jpg', annotated_frame)
                #plt.imshow(annotated_frame); 
                #plt.show()
                detections = [detection for detection in zip(xyxy.tolist(), logits.tolist(), phrases) if detection[-1] in OBJECTIVE_PROMPTS]
                xyxy, logits, phrases = zip(*non_maximum_suppression(detections, iou_threshold=0.5))

                nms_frame, xyxy = annotate_nms(image_source=og_image, xyxy=np.array(list(xyxy)), logits=torch.as_tensor(logits), phrases=phrases)
                xyxy, logits, phrases = np.array(list(xyxy)), list(logits), list(phrases)
                #plt.imsave('after_nms.jpg', nms_frame)
                #print("save nms")
                object_labels, object_poses, seg_img, obj_pixel_location, xyxy, ins_seg_add = self.cluster.sam_bbox_prompt(og_image.copy(), depth, self.camera_matrix, labels=phrases, bbox=xyxy, camera_pose = camera_pose)
                plt.imshow(seg_img);
                plt.show()

                depth_map_path = "perception_data/black_paper/0/depth_1698292165.npy"
                depth_map=np.load(depth_map_path)
                
                #plt.imshow(depth_map)
                #plt.axis('off')
                #plt.show()
 
                #visualize_scene_pc(np.asarray(ins_seg_add), depth_map, self.camera_matrix)

                for id, (pose, pixel_location) in enumerate(zip(object_poses, obj_pixel_location)):
                    obj_pose_world=camera_pose.dot(pose)
                    xy = str(np.round(obj_pose_world[:2, 3], decimals=3))    
                    cv2.putText(seg_img, f'{id}:{xy}', pixel_location, cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
                
                ## Write to knowledge base
                for j in range(len(object_poses)):
                    world_pose = camera_pose.dot(object_poses[j])
                    self.knowledge_base[f'object_{j}'] = {
                        'object_type': map_prompt_to_object(object_labels[j]),
                        'translation': pose[:3,3].tolist(),
                        'rotation': pose[:3,:3].tolist(),
                        'bbox': xyxy[j].astype(int).tolist()
                    }

                # Get ground truth data
                print(f'{row["root"]}/image_{row["data_id"]}.jpg')
                #plt.axis("off")
                plt.imsave('final_prediction.jpg', seg_img)
                plt.imshow(seg_img)
                plt.show()
                bb_map = []
                for i in range(max(len(labels),len(xyxy))):
                    try:
                        bb_map.append(
                        {
                        'ground_truth': labels[i][1:], 
                        'predicted_bb': None,
                        'object_type': labels[i][0],
                        'predicted_object_type': None,
                        'iou': 0
                        })
                    except IndexError:
                        bb_map.append(
                        {
                        'ground_truth': None, 
                        'predicted_bb': None,
                        'object_type': 'None',
                        'predicted_object_type': None,
                        'iou': 0
                        })
                
                for i in range(len(labels)):
                    for j in range(len(xyxy)):
                        bb=xyxy[j]
                        iou = iou_bbox(labels[i][1:], bb)
                        if bb_map[i]['object_type'] == 'plate':
                            plate_ious.append(iou)
                        bb_map[i]['predicted_object_type'] = map_prompt_to_object(phrases[j])
                        if iou > bb_map[i]['iou']:
                            bb_map[i]['predicted_bb'] = bb
                            bb_map[i]['iou'] = iou

                    cv2.rectangle(og_image.copy(), (labels[i][1], labels[i][2]), (labels[i][3], labels[i][4]), (0,0,255), 2)


                for i in range(len(labels)):
                    if bb_map[i]['predicted_bb'] is not None:
                        bb_idx = np.where((xyxy == bb_map[i]['predicted_bb']).all(axis=1))[0][0]
                        world_pose = camera_pose.dot(object_poses[bb_idx])[:3,3]
                        position_errs = [np.linalg.norm(world_pose-pose[1].matrix[:3,3]) for pose in ground_truth_positions]
                        min_idx = np.argmin(position_errs)
                        gt_id, _ = ground_truth_positions[min_idx]
                        ht = self.objects[self.objects['ID']==int(gt_id)]['height(m)']
                        world_pose[2] -= ht/2
                        position_errs = [np.linalg.norm(world_pose-pose[1].matrix[:3,3]) for pose in ground_truth_positions]
                        bb_map[i]['mae'] = position_errs[min_idx]
                        
                        if int(gt_id) == 2:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            object_2.append(position_errs[min_idx]) 
                        elif int(gt_id) == 5:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            object_5.append(position_errs[min_idx])
                        elif int(gt_id) == 14:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            object_14.append(position_errs[min_idx])
                        elif int(gt_id) == 17:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            object_17.append(position_errs[min_idx])
                        elif int(gt_id) == 23:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            object_23.append(position_errs[min_idx])


                history += bb_map


        print(f'object_2: {object_2}')
        print(f'object_5: {object_5}')
        print(f'object_14: {object_14}')
        print(f'object_17: {object_17}')
        print(f'object_23: {object_23}')

        # print(sum(object_2)/len(object_2))
        # print(sum(object_5)/len(object_5))
        # print(sum(object_14)/len(object_14))
        # print(sum(object_17)/len(object_17))
        # print(sum(object_23)/len(object_23))

        df = pd.DataFrame(history)
        #aps = calculate_class_average_precision(df['iou'], df['object_type'])
        mae = df['mae'].mean()
        #plate_ious = np.array(plate_ious)
        #plate_ious = plate_ious[plate_ious>0.2]
        #breakpoint()
        #print(aps)
        print(mae)
        # with open(f'{self.data_dir}/knowledge_base.json','w+') as f:
        #     json.dump(self.knowledge_base, f)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    #perception_evaluator = PerceptionEval(['bottle', 'plate', 'pod'])
    perception_evaluator = PerceptionEval(['bowl', 'lemon', 'apple'])
    perception_evaluator.evaluate()
