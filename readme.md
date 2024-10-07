# ROS FoundationPose

## About

This repository is a ROS1 implementation of [Foundation Pose](https://github.com/NVlabs/FoundationPose), useful for model-based 6DoF pose tracking. This was originally developed by Bowen et. al., describled in their paper titled "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects".

RGB-D images data is obtained from an Intel RealSense Depth camera using the [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) wrapper.

The binary mask of the first frame, as required by the model-based implementation of FoundationPose (refer to the original paper), is generated using [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), from IDEA-Research. This uses Grounding-DINO, developed by IDEA-Research and Segment Anything Model (SAM), developed by Meta. 

## Setup

This repository consists of two directories, the FoundationPose_ROS and the Grounded-Segment-Anything. Once the repository has been cloned, weights for each model need to be downloaded.

The weights for FoundationPose can be downloaded from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing). The two folders downloaded should be placed as-is in `FoundationPose_ROS/weights/`. 

The weights for GroundedSAM can be downloaded using the following commands:
```bash
cd Grounded-Segment-Anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Due to dependecy clashes, FoundationPose and GroundedSAM ROS nodes work in separate docker containers. You can pull working containers for each from:
```
docker pull ghcr.io/shubho-upenn/ros_fp_new_env:latest	## FoundationPose Container Image
docker pull ghcr.io/shubho-upenn/gsa_ros:latest		## Grounded-Segment-Anything Container Image
```
### Prerequisites
1. GPU with Cuda >= 11.7
2. ROS Noetic

## Implementation

### 1. Run the docker containers 

To run the docker containers, open two separate terminals. In the first terminal run the GroundedSAM ROS container using:
```bash
cd Grounded-Segment-Anything
bash run_gsa_ros.sh
```

In the other terminal run the FoundationPose ROS container using:
```bash
cd FoundationPose_ROS/docker
bash run_container_new_env.sh
```

### 2. Launch the RealSense Node
Launch the RealSense node in a new terminal on the host computer. Be sure to set `align_depth:=true`. 

### 3. Run the GroundedSAM ROS node
In the gsa_ros (GroundedSAM) container do the following to run the ROS GroundedSAM:
```bash
cd Grounded-Segment-Anything
python3 ros_gsam.py
```
Enter the object that you want to track to enable the GroundedSAM to detect and segment the object in the image and press enter. Accurate and general inputs such as 'mustard bottle' or 'orange hand drill' (as in examples) will work well.

### 4. Run the FoundationPose ROS node
Before running the command, place the `.obj` file of the object you want to track in the `mesh` directory.
Then, in the ros_fp_new_env (FoundationPose) container, do the following to run the ROS Foundation Pose tracking:
```bash
conda activate test_env
python3 run_ROS.py -in path/to/obj/file
```
You should see the object being tracked.

## Acknowldegement
This repository is based on the work and takes major components from FoundationPose developed by Bowen et. al. (NVLabs) and GroundedSAM from IDEA-Research and Meta.
The Licence for FoundationPose can be found in ./FoundationPose_ROS/LICENCE.
The Licence for Grounded-SAM can be found in ./Grounded-Segment-Anything/LICENCE.
The Licence for segment-anything can be found in ./Grounded-Segment-Anything/segment_anything/LICENCE.
The original readme.md of each of these repos can also be found in their respective aforementioned directories.

Bibtex Citation for FoundationPose:
```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

Bibtex Citation for Grounded-SAM (GroundingDINO and SegmentAnything):
```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

