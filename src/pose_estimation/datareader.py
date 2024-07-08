# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.




from Utils import *
import json,os,sys
import open3d
#from local_utils.config_utils import parse_config_utils
from segmentation import *
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
class OrganaReader:
  def __init__(self, base_dir, downscale=1, shorter_side=None, zfar=np.inf):
    self.base_dir = base_dir
    self.downscale = downscale
    self.zfar = zfar
    self.dataset_name = 'organa'
    self.ob_ids = [1698287651, 1698287496, 1698287390, 1698287525, 1698441059, 1698442292, 1698442659, 1698441882]
    #self.load_symmetry_tfs()
    #self.color_files = sorted(glob.glob(f"{self.base_dir}/**/**/*.jpg"))
       # specify the path to the rgb images here and makr sure the images are in the right format
    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*.jpg"))
    # camera intrinsic matrix for the organa dataset
    self.K = np.array([[729.42260742,   0.        , 617.55908203],
                                [  0.        , 729.42260742, 359.8135376 ],
                            [  0.        ,   0.        ,   1.        ]])
    self.color_sub = rospy.Subscriber('/zedm/zed_node/rgb/image_rect_color', Image, self.color_callback)
    self.depth_sub = rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image, self.depth_callback)
    self.camera_info_sub = rospy.Subscriber('/zedm/zed_node/rgb/camera_info', CameraInfo, self.camera_info_callback)
    # camera intrinsic matrix for the foundationpose demo dataset
 #     self.K = np.array([[3.195820007324218750e+02, 0.000000000000000000e+00, 3.202149847676955687e+02],
 #  [0.000000000000000000e+00, 4.171186828613281250e+02, 2.443486680871046701e+02],
 #  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
  # camera intrinsic matrix for the RGBD object dataset
 #     self.K = np.array([
 # [510.0, 0, 319.5],
 # [0, 525.0, 238.5],
 # [0, 0, 1]
 # ])
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.jpg','')
      self.id_strs.append(id_str)
    self.H,self.W = 2560, 720

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)
    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale
    # self.gt_pose_files = sorted(glob.glob(f'{self.base_dir}/gt_poses/*.tf'))
    # self.depth_paths = sorted(glob.glob(f'{base_dir}/**/**/depth_*.npy', recursive=True))
    # self.camera_pose_paths = sorted(glob.glob(f'{base_dir}/**/**/camera_pose_*.npy', recursive=True))
    self.depth_paths = sorted(glob.glob(f'{self.base_dir}/depth/*.npy', recursive=True))
    self.labels = glob.glob(f'{base_dir}/**/**/*.xml', recursive=True)
    #self.metadata = pd.read_csv(f'{base_dir}/metadata.csv')

    self.color_dir = f'{base_dir}/rgb'
    self.depth_dir = f'{base_dir}/depth'
    os.makedirs(self.color_dir, exist_ok=True)
    os.makedirs(self.depth_dir, exist_ok=True)
    self.image_counter = 0
    self.color_image = None
    self.depth_image = None
    self.bridge = CvBridge()
    

  def __len__(self):
    return len(self.color_files)
  def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)


  def color_callback(self, msg):
      try:
          self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
          if self.color_image is not None and self.depth_image is not None:
              self.process_images()
      except CvBridgeError as e:
          rospy.logerr(e)


  def depth_callback(self, msg):
      try:
          self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
          if self.color_image is not None and self.depth_image is not None:
              self.process_images()
      except CvBridgeError as e:
          rospy.logerr(e)
  def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)
      
  def process_images(self):
        # Process images for pose estimation
        if self.color_image is not None and self.depth_image is not None:
            # Save color image
            color_image_path = os.path.join(self.color_dir, f'{self.image_counter:06d}.jpg')
            cv2.imwrite(color_image_path, self.color_image)
            # Save depth image
            depth_image_path = os.path.join(self.depth_dir, f'{self.image_counter:06d}.npy')
            np.save(depth_image_path, self.depth_image)

            rospy.loginfo(f'Saved color image to {color_image_path}')
            rospy.loginfo(f'Saved depth image to {depth_image_path}')

            # Increment the image counter
            self.image_counter += 1

            print("Images received and processed")
  

  def get_camera_pose(self,i):
    try:
      pose = np.load(self.camera_pose_paths[i])
      return pose
    except:
      logging.info("Camera pose not found, return None")

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None
  # def get_color(self,i):
  #    # read from the image topic
  #   self.color_image = None
  #   self.depth_image = None
  #   self.bridge = CvBridge()
  #   rospy.init_node('image_listener', anonymous=True)
  #   rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
  #   rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
  #   rospy.spin()
  #   return self.color_image


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color
  def generate_mask(self):
    mask_generator = MaskGenerator(base_dir=self.base_dir)
    mask_generator.generation()


    # Load the bounding box coordinates

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask
  # def get_depth(self,i):
  #   depth = np.load(self.depth_paths[i])
  #   depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
  #   depth[(depth<0.1) | (depth>=self.zfar)] = 0

  #   if depth.ndim == 3:
  #   # Extract the first channel
  #     depth = depth[:, :, 0]
  # # Now, depth should be 2-dimensional
  # #  Proceed with the erode_depth function
  #   depth = erode_depth(depth, radius=2, device='cuda')
  #   return depth
  def get_depth(self,i):
    #depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    # depth is in npy format


    depth = np.load(self.depth_paths[i])
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.1) | (depth>=self.zfar)] = 0
    if depth.ndim == 3:
    # Extract the first channel
      depth = depth[:, :, 0]
    depth = erode_depth(depth, radius=2, device='cuda')
    # return depth
    return depth


  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map
  def get_occ_mask(self,i):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.H,self.W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)
    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)
    occ_mask = cv2.resize(occ_mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return occ_mask.astype(np.uint8)
  def get_gt_mesh(self):
    pass
  def evaluate(self):
    # compare the estimated poses with the ground truth poses in the dataset .tf
    pass
  def get_rgbd(self,i):
    color_raw = open3d.io.read_image(self.color_files[i])
    depth_raw = open3d.io.read_image(self.depth_paths[i])
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    return rgbd_image
  def get_point_cloud(self,i):
    rgbd_image = self.get_rgbd(i)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.K)
    return pcd
 
 
