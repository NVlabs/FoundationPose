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
from segmentation import MaskGenerator

BOP_LIST = ['lmo','tless','ycbv','hb','tudl','icbin','itodd']
BOP_DIR = os.getenv('BOP_DIR')



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
    self.color_files = sorted(glob.glob(f"{self.base_dir}/*.jpg"))
    self.depth_files = sorted(glob.glob(f"{self.base_dir}/depth_*.npy"))
    self.camera_pose_files = sorted(glob.glob(f"{self.base_dir}/camera_pose_*.npy"))
    self.bounding_box_files = sorted(glob.glob(f"{self.base_dir}/bounding_box_*.npy"))
    self.labels = glob.glob(f'{base_dir}/*.xml', recursive=True)
    self.gt_pose_files = sorted(glob.glob(f'{self.base_dir}/*.tf'))
    self.mask_dir = f"{self.base_dir}/masks"
    self.gt_pose_beaker_250ml_files = sorted(glob.glob(f"{self.base_dir}/5.tf"))
    self.gt_pose_conical_flask_500ml_files = sorted(glob.glob(f"{self.base_dir}/17.tf"))
    self.gt_pose_conical_flask_250ml_files = sorted(glob.glob(f"{self.base_dir}/14.tf"))
    self.gt_pose_beaker_30ml_files = sorted(glob.glob(f"{self.base_dir}/2.tf"))
    self.est_pose_beaker_250ml_files = sorted(glob.glob(f"{self.base_dir}/estimated_*_pose_beaker_250ml_*.npy"))
    self.est_pose_conical_flask_500ml_files = sorted(glob.glob(f"{self.base_dir}/estimated_*_pose_conical_flask_500ml_*.npy"))
    self.est_pose_conical_flask_250ml_files = sorted(glob.glob(f"{self.base_dir}/estimated_*_pose_conical_flask_250ml_*.npy"))
    self.est_pose_beaker_30ml_files = sorted(glob.glob(f"{self.base_dir}/estimated_*_pose_beaker_30ml_*.npy"))
    self.est_pose_files = sorted(glob.glob(f"{self.base_dir}/estimated_*.npy"))
    print(self.color_files)
    # camera intrinsic matrix for the organa dataset
    self.K = np.array([[729.42260742,   0.        , 617.55908203],
                                [  0.        , 729.42260742, 359.8135376 ],
                            [  0.        ,   0.        ,   1.        ]])

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
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]
    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)
    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale
    self.gt_pose_files = sorted(glob.glob(f'{self.base_dir}/*.tf'))
    # #
    # self.depth_paths = sorted(glob.glob(f'{base_dir}/depth_*.npy', recursive=True))
    # self.camera_pose_paths = sorted(glob.glob(f'{base_dir}/camera_pose_*.npy', recursive=True))
    

    self.labels = glob.glob(f'{base_dir}/*.xml', recursive=True)
    #self.metadata = pd.read_csv(f'{base_dir}/metadata.csv')

  def __len__(self):
    return len(self.color_files)
  
  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None
  def get_color(self,i):
    #print(self.color_files[i])
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color
  def generate_mask(self, mesh_name, frame_id):
    mask_generator = MaskGenerator(base_dir=self.base_dir, mesh_name=mesh_name, frame_id=frame_id)
    mask_generator.generation()
    print("Mask generation done")
  def get_mask(self,i, mesh_name, frame_id):
    mask = cv2.imread(f'{self.mask_dir}' + '/' + mesh_name + '_mask'+f'_{frame_id}.png')
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
    print(self.depth_files[i])
    depth = np.load(self.depth_files[i])
    #depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3

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
  def get_point_cloud(self,i):
    rgbd_image = self.get_rgbd(i)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.K)
    return pcd
    


class YcbineoatReader:
  def __init__(self,video_dir, downscale=1, shorter_side=None, zfar=np.inf):
    self.video_dir = video_dir
    self.downscale = downscale
    self.zfar = zfar
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
    self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.jpg','')
      self.id_strs.append(id_str)
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_pose_files = sorted(glob.glob(f'{self.video_dir}/annotated_poses/*'))

    self.videoname_to_object = {
      'bleach0': "021_bleach_cleanser",
      'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
    }

  def get_video_name(self):
    return self.video_dir.split('/')[-1]

  def __len__(self):
    return len(self.color_files)

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask

  def get_depth(self,i):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.1) | (depth>=self.zfar)] = 0
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
    ob_name = self.videoname_to_object[self.get_video_name()]
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj')
    return mesh

class BopBaseReader:
  def __init__(self, base_dir, zfar=np.inf, resize=1):
    self.base_dir = base_dir
    self.resize = resize
    self.dataset_name = None
    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*"))
    if len(self.color_files)==0:
      self.color_files = sorted(glob.glob(f"{self.base_dir}/gray/*"))
    self.zfar = zfar

    self.K_table = {}
    with open(f'{self.base_dir}/scene_camera.json','r') as ff:
      info = json.load(ff)
    for k in info:
      self.K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3,3)
      self.bop_depth_scale = info[k]['depth_scale']

    if os.path.exists(f'{self.base_dir}/scene_gt.json'):
      with open(f'{self.base_dir}/scene_gt.json','r') as ff:
        self.scene_gt = json.load(ff)
      self.scene_gt = copy.deepcopy(self.scene_gt)   # Release file handle to be pickle-able by joblib
      assert len(self.scene_gt)==len(self.color_files)
    else:
      self.scene_gt = None

    self.make_id_strs()


  def make_scene_ob_ids_dict(self):
    with open(f'{BOP_DIR}/{self.dataset_name}/test_targets_bop19.json','r') as ff:
      self.scene_ob_ids_dict = {}
      data = json.load(ff)
      for d in data:
        if d['scene_id']==self.get_video_id():
          id_str = f"{d['im_id']:06d}"
          if id_str not in self.scene_ob_ids_dict:
            self.scene_ob_ids_dict[id_str] = []
          self.scene_ob_ids_dict[id_str] += [d['obj_id']]*d['inst_count']


  def get_K(self, i_frame):
    K = self.K_table[self.id_strs[i_frame]]
    if self.resize!=1:
      K[:2,:2] *= self.resize
    return K


  def get_video_dir(self):
    video_id = int(self.base_dir.rstrip('/').split('/')[-1])
    return video_id

  def make_id_strs(self):
    self.id_strs = []
    for i in range(len(self.color_files)):
      name = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(name)


  def get_instance_ids_in_image(self, i_frame:int):
    ob_ids = []
    if self.scene_gt is not None:
      name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
      for k in self.scene_gt[str(name)]:
        ob_ids.append(k['obj_id'])
    elif self.scene_ob_ids_dict is not None:
      return np.array(self.scene_ob_ids_dict[self.id_strs[i_frame]])
    else:
      mask_dir = os.path.dirname(self.color_files[0]).replace('rgb','mask_visib')
      id_str = self.id_strs[i_frame]
      mask_files = sorted(glob.glob(f'{mask_dir}/{id_str}_*.jpg'))
      ob_ids = []
      for mask_file in mask_files:
        ob_id = int(os.path.basename(mask_file).split('.')[0].split('_')[1])
        ob_ids.append(ob_id)
    ob_ids = np.asarray(ob_ids)
    return ob_ids


  def get_gt_mesh_file(self, ob_id):
    raise RuntimeError("You should override this")


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])
    if len(color.shape)==2:
      color = np.tile(color[...,None], (1,1,3))  # Gray to RGB
    if self.resize!=1:
      color = cv2.resize(color, fx=self.resize, fy=self.resize, dsize=None)
    return color


  def get_depth(self,i, filled=False):
    if filled:
      depth_file = self.color_files[i].replace('rgb','depth_filled')
      depth_file = f'{os.path.dirname(depth_file)}/0{os.path.basename(depth_file)}'
      depth = cv2.imread(depth_file,-1)/1e3
    else:
      depth_file = self.color_files[i].replace('rgb','depth').replace('gray','depth')
      depth = cv2.imread(depth_file,-1)*1e-3*self.bop_depth_scale
    if self.resize!=1:
      depth = cv2.resize(depth, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    depth[depth<0.1] = 0
    depth[depth>self.zfar] = 0
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.get_K(i))
    return xyz_map


  def get_mask(self, i_frame:int, ob_id:int, type='mask_visib'):
    '''
    @type: mask_visib (only visible part) / mask (projected mask from whole model)
    '''
    pos = 0
    name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
    if self.scene_gt is not None:
      for k in self.scene_gt[str(name)]:
        if k['obj_id']==ob_id:
          break
        pos += 1
      mask_file = f'{self.base_dir}/{type}/{name:06d}_{pos:06d}.jpg'
      if not os.path.exists(mask_file):
        logging.info(f'{mask_file} not found')
        return None
    else:
      # mask_dir = os.path.dirname(self.color_files[0]).replace('rgb',type)
      # mask_file = f'{mask_dir}/{self.id_strs[i_frame]}_{ob_id:06d}.jpg'
      raise RuntimeError
    mask = cv2.imread(mask_file, -1)
    if self.resize!=1:
      mask = cv2.resize(mask, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    return mask>0


  def get_gt_mesh(self, ob_id:int):
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file)
    mesh.vertices *= 1e-3
    return mesh


  def get_model_diameter(self, ob_id):
    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    return info[str(ob_id)]['diameter']/1e3



  def get_gt_poses(self, i_frame, ob_id):
    gt_poses = []
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        gt_poses.append(cur)
    return np.asarray(gt_poses).reshape(-1,4,4)


  def get_gt_pose(self, i_frame:int, ob_id, mask=None, use_my_correction=False):
    ob_in_cam = np.eye(4)
    best_iou = -np.inf
    best_gt_mask = None
    name = int(self.id_strs[i_frame])
    for i_k, k in enumerate(self.scene_gt[str(name)]):
      if k['obj_id']==ob_id:
        cur = np.eye(4)
        cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(k['cam_t_m2c'])/1e3
        if mask is not None:  # When multi-instance exists, use mask to determine which one
          gt_mask = cv2.imread(f'{self.base_dir}/mask_visib/{self.id_strs[i_frame]}_{i_k:06d}.jpg', -1).astype(bool)
          intersect = (gt_mask*mask).astype(bool)
          union = (gt_mask+mask).astype(bool)
          iou = float(intersect.sum())/union.sum()
          if iou>best_iou:
            best_iou = iou
            best_gt_mask = gt_mask
            ob_in_cam = cur
        else:
          ob_in_cam = cur
          break


    if use_my_correction:
      if 'ycb' in self.base_dir.lower() and 'train_real' in self.color_files[i_frame]:
        video_id = self.get_video_id()
        if ob_id==1:
          if video_id in [12,13,14,17,24]:
            ob_in_cam = ob_in_cam@self.symmetry_tfs[ob_id][1]
    return ob_in_cam


  def load_symmetry_tfs(self):
    dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
    info_file = f'{dir}/models_info.json'
    with open(info_file,'r') as ff:
      info = json.load(ff)
    self.symmetry_tfs = {}
    self.symmetry_info_table = {}
    for ob_id in self.ob_ids:
      self.symmetry_info_table[ob_id] = info[str(ob_id)]
      self.symmetry_tfs[ob_id] = symmetry_tfs_from_info(info[str(ob_id)], rot_angle_discrete=5)
    self.geometry_symmetry_info_table = copy.deepcopy(self.symmetry_info_table)


  def get_video_id(self):
    return int(self.base_dir.split('/')[-1])


class LinemodOcclusionReader(BopBaseReader):
  def __init__(self,base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD-O/lmo_test_all/test/000002', zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lmo'
    self.K = list(self.K_table.values())[0]
    self.ob_ids = [1,5,6,8,9,10,11,12]
    self.ob_id_to_names = {
      1: 'ape',
      2: 'benchvise',
      3: 'bowl',
      4: 'camera',
      5: 'water_pour',
      6: 'cat',
      7: 'cup',
      8: 'driller',
      9: 'duck',
      10: 'eggbox',
      11: 'glue',
      12: 'holepuncher',
      13: 'iron',
      14: 'lamp',
      15: 'phone',
    }
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_dir = f'{BOP_DIR}/{self.dataset_name}/models/obj_{ob_id:06d}.ply'
    return mesh_dir



class LinemodReader(LinemodOcclusionReader):
  def __init__(self, base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/lm_test_all/test/000001', zfar=np.inf, split=None):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'lm'
    if split is not None:  # train/test
      with open(f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/Linemod_preprocessed/data/{self.get_video_id():02d}/{split}.txt','r') as ff:
        lines = ff.read().splitlines()
      self.color_files = []
      for line in lines:
        id = int(line)
        self.color_files.append(f'{self.base_dir}/rgb/{id:06d}.jpg')
      self.make_id_strs()

    self.ob_ids = np.setdiff1d(np.arange(1,16), np.array([7,3])).tolist()  # Exclude bowl and mug
    self.load_symmetry_tfs()


  def get_gt_mesh_file(self, ob_id):
    root = self.base_dir
    while 1:
      if os.path.exists(f'{root}/lm_models'):
        mesh_dir = f'{root}/lm_models/models/obj_{ob_id:06d}.ply'
        break
      else:
        root = os.path.abspath(f'{root}/../')
    return mesh_dir


  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh


class YcbVideoReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'ycbv'
    self.K = list(self.K_table.values())[0]

    self.make_id_strs()

    self.ob_ids = np.arange(1,22).astype(int).tolist()
    YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
    names = sorted(os.listdir(f'{YCB_VIDEO_DIR}/models/'))
    self.ob_id_to_names = {}
    self.name_to_ob_id = {}
    for i,ob_id in enumerate(self.ob_ids):
      self.ob_id_to_names[ob_id] = names[i]
      self.name_to_ob_id[names[i]] = ob_id

    if 'BOP' not in self.base_dir:
      with open(f'{self.base_dir}/../../keyframe.txt','r') as ff:
        self.keyframe_lines = ff.read().splitlines()

    self.load_symmetry_tfs()
    for ob_id in self.ob_ids:
      if ob_id in [1,4,6,18]:   # Cylinder
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
              {'axis':[0,0,1], 'offset':[0,0,0]},
            ],
          'symmetries_discrete': euler_matrix(0, np.pi, 0).reshape(1,4,4).tolist(),
          }
      elif ob_id in [13]:
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_continuous': [
              {'axis':[0,0,1], 'offset':[0,0,0]},
            ],
          }
      elif ob_id in [2,3,9,21]:   # Rectangle box
        tfs = []
        for rz in [0, np.pi]:
          for rx in [0,np.pi]:
            for ry in [0,np.pi]:
              tfs.append(euler_matrix(rx, ry, rz))
        self.geometry_symmetry_info_table[ob_id] = {
          'symmetries_discrete': np.asarray(tfs).reshape(-1,4,4).tolist(),
          }
      else:
        pass

  def get_gt_mesh_file(self, ob_id):
    if 'BOP' in self.base_dir:
      mesh_file = os.path.abspath(f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply')
    else:
      mesh_file = f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id:int, get_posecnn_version=False):
    if get_posecnn_version:
      YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
      mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{self.ob_id_to_names[ob_id]}/textured_simple.obj')
      return mesh
    mesh_file = self.get_gt_mesh_file(ob_id)
    mesh = trimesh.load(mesh_file, process=False)
    mesh.vertices *= 1e-3
    tex_file = mesh_file.replace('.ply','.jpg')
    if os.path.exists(tex_file):
      from PIL import Image
      im = Image.open(tex_file)
      uv = mesh.visual.uv
      material = trimesh.visual.texture.SimpleMaterial(image=im)
      color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
      mesh.visual = color_visuals
    return mesh


  def get_reconstructed_mesh(self, ob_id, ref_view_dir):
    mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
    return mesh


  def get_transform_reconstructed_to_gt_model(self, ob_id):
    out = np.eye(4)
    return out


  def get_visible_cloud(self, ob_id):
    file = os.path.abspath(f'{self.base_dir}/../../models/{self.ob_id_to_names[ob_id]}/visible_cloud.ply')
    pcd = o3d.io.read_point_cloud(file)
    return pcd


  def is_keyframe(self, i):
    color_file = self.color_files[i]
    video_id = self.get_video_id()
    frame_id = int(os.path.basename(color_file).split('.')[0])
    key = f'{video_id:04d}/{frame_id:06d}'
    return (key in self.keyframe_lines)



class TlessReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tless'

    self.ob_ids = np.arange(1,31).astype(int).tolist()
    self.load_symmetry_tfs()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../models_cad/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_mesh(self, ob_id):
    mesh = trimesh.load(self.get_gt_mesh_file(ob_id))
    mesh.vertices *= 1e-3
    mesh = trimesh_add_pure_colored_texture(mesh, color=np.ones((3))*200)
    return mesh


class HomebrewedReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'hb'
    self.ob_ids = np.arange(1,34).astype(int).tolist()
    self.load_symmetry_tfs()
    self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../hb_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


  def get_gt_pose(self, i_frame:int, ob_id, use_my_correction=False):
    logging.info("WARN HomeBrewed doesn't have GT pose")
    return np.eye(4)



class ItoddReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'itodd'
    self.make_id_strs()

    self.ob_ids = np.arange(1,29).astype(int).tolist()
    self.load_symmetry_tfs()
    self.make_scene_ob_ids_dict()


  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../itodd_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class IcbinReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'icbin'
    self.ob_ids = np.arange(1,3).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../icbin_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


class TudlReader(BopBaseReader):
  def __init__(self, base_dir, zfar=np.inf):
    super().__init__(base_dir, zfar=zfar)
    self.dataset_name = 'tudl'
    self.ob_ids = np.arange(1,4).astype(int).tolist()
    self.load_symmetry_tfs()

  def get_gt_mesh_file(self, ob_id):
    mesh_file = f'{self.base_dir}/../../../tudl_models/models/obj_{ob_id:06d}.ply'
    return mesh_file


