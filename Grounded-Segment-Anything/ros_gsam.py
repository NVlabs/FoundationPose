#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image as Img

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image_from_array(image_array):
    # Convert numpy array to PIL Image
    image_pil = Img.fromarray(image_array).convert("RGB")

    # Define transform
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=2000),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    # Apply transform
    image, _ = transform(image_pil, None)  # 3, h, w
    # print(image.shape)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def publish_mask_image(masks):
    bridge = CvBridge()
    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert mask to binary image
        mask_img = Img.fromarray(mask_np[0])  # Convert numpy array to PIL Image
        mask_img_cv = np.array(mask_img)  # Convert PIL Image to numpy array
        mask_msg = bridge.cv2_to_imgmsg(mask_img_cv, encoding="mono8")  # Convert OpenCV image to ROS Image message
        mask_publisher.publish(mask_msg)


img_num = 0

def image_callback(msg):

    global img_num

    # try:
    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    image_cv = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # rospy.loginfo("Recieved image")

    # Display the image
    # cv2.imshow("RealSense RGB Image", image_rgb)
    # cv2.waitKey(1)  # Refresh every 1 millisecond

    if img_num == 0:
        img_num = img_num + 1
        # cv2.imwrite("abc.png", image_rgb)
        ### RUN GROUNDED SAM ###
        grounded_sam(image_cv)
        rospy.loginfo("GROUNDED SAM OUTPUT DONE: Publishied Mask for First Frame")

    
    # except Exception as e:
    #     print(e)


def grounded_sam(image):
    
    # cfg
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" ## args.config  # change the path of the model config file
    grounded_checkpoint = "groundingdino_swint_ogc.pth"                           ## args.grounded_checkpoint  # change the path of the model
    sam_version = "vit_h"                                                         ## args.sam_version
    sam_checkpoint = "sam_vit_h_4b8939.pth"                                       ## args.sam_checkpoint
    # sam_hq_checkpoint = args.sam_hq_checkpoint
    # use_sam_hq = args.use_sam_hq
    # image_path = image

    text_input = input("ENTER WHAT OBJECT TO TRACK: ")  ## INPUT                                      ## args.text_prompt
    
    if text_input:
        text_prompt = text_input
    
    else:
        text_prompt = "mustard bottle"

    output_dir = "model_outputs"
    box_threshold = 0.3
    text_threshold = 0.25
    
    if torch.cuda.is_available():
        device = "cuda"
        print('DEVICE - ', device)
    else:
        device = "cpu"

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load image
    image_pil, image_1 = load_image_from_array(image)
    # print("image - ", image)
    # print("image_pil - ", image_pil)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image_1, text_prompt, box_threshold, text_threshold, device=device)

    model.to('cpu')
    torch.cuda.empty_cache()
    # model.to('cpu')

    # initialize SAM
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    torch.cuda.empty_cache()

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "000000.png"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
    publish_mask_image(masks)

    torch.cuda.empty_cache()
    del predictor
    torch.cuda.empty_cache()



def main():
    global mask_publisher
    rospy.init_node('realsense_subscriber_gsam', anonymous=True)
    mask_publisher = rospy.Publisher("/mask_image", Image, queue_size=10, latch=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.spin()

if __name__ == "__main__":
    main()