# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import *
from pytorch3d.transforms import so3_log_map,so3_exp_map,se3_exp_map

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mae = lambda x, y: (torch.abs(x - y)).mean()
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


class FeatureArray(nn.Module):
    """
    Per-frame corrective latent code.
    """

    def __init__(self, num_frames, num_channels):
        super().__init__()

        self.num_frames = num_frames
        self.num_channels = num_channels

        self.data = nn.parameter.Parameter(torch.normal(0,1,size=[num_frames, num_channels]).float(), requires_grad=True)
        self.register_parameter('data',self.data)


    def __call__(self, ids):
        return self.data[ids]


class PoseArray(nn.Module):
    def __init__(self, num_frames,max_trans,max_rot):
        super().__init__()
        self.num_frames = num_frames
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.data = nn.parameter.Parameter(torch.zeros([num_frames, 6]).float(), requires_grad=True)
        self.register_parameter('data',self.data)


    def get_matrices(self,ids):
        if not torch.is_tensor(ids):
          ids = torch.tensor(ids).long()
        theta = torch.tanh(self.data)
        trans = theta[:,:3] * self.max_trans
        rot = theta[:,3:6] * self.max_rot/180.0*np.pi
        Ts_data = se3_exp_map(torch.cat((trans,rot),dim=-1)).permute(0,2,1)
        Ts = torch.eye(4, device=self.data.device).reshape(1,4,4).repeat(len(ids),1,1)
        mask = ids!=0
        Ts[mask] = Ts_data[ids[mask]]
        return Ts



class SHEncoder(nn.Module):
    '''Spherical encoding
    '''
    def __init__(self, input_dim=3, degree=4):

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, cfg, i=0, octree_m=None):
    if i == -1:
        return nn.Identity(), 3
    elif i==0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }

        embed = Embedder(**embed_kwargs)
        out_dim = embed.out_dim
    elif i==1:
        from mycuda.torch_ngp_grid_encoder.grid import GridEncoder
        embed = GridEncoder(input_dim=3, n_levels=cfg['num_levels'], log2_hashmap_size=cfg['log2_hashmap_size'], desired_resolution=cfg['finest_res'], base_resolution=cfg['base_res'], level_dim=cfg['feature_grid_dim'])
        print(embed)
        out_dim = embed.out_dim
    elif i==2:
        embed = SHEncoder(degree=cfg['multires_views'])
        out_dim = embed.out_dim
    return embed, out_dim



def mesh_to_real_world(mesh,pose_offset,translation,sc_factor):
    '''
    @pose_offset: optimized delta pose of the first frame. Usually it's identity
    '''
    mesh.vertices = mesh.vertices/sc_factor - np.array(translation).reshape(1,3)
    mesh.apply_transform(pose_offset)
    return mesh


def get_optimized_poses_in_real_world(poses_normalized, pose_array, sc_factor, translation):
    '''
    @poses_normalized: np array, cam_in_ob (opengl convention), normalized to [-1,1] and centered
    @pose_array: PoseArray, delta poses
    Return:
        cam_in_ob, real-world unit, opencv convention
    '''
    original_poses = poses_normalized.copy()
    original_poses[:, :3, 3] /= sc_factor   # To true world scale
    original_poses[:, :3, 3] -= translation

    # Apply pose transformation
    tf = pose_array.get_matrices(np.arange(len(poses_normalized))).reshape(-1,4,4).data.cpu().numpy()
    optimized_poses = tf@poses_normalized

    optimized_poses = np.array(optimized_poses).astype(np.float32)
    optimized_poses[:, :3, 3] /= sc_factor
    optimized_poses[:, :3, 3] -= translation

    original_init_ob_in_cam = optimized_poses[0].copy()
    offset = np.linalg.inv(original_init_ob_in_cam)@original_poses[0]
    for i in range(len(optimized_poses)):
      new_ob_in_cam = optimized_poses[i]@offset
      optimized_poses[i] = new_ob_in_cam
      optimized_poses[i] = optimized_poses[i]@glcam_in_cvcam

    return optimized_poses,offset

def preprocess_data(rgbs,depths,masks,normal_maps,poses,sc_factor,translation):
    '''
    @rgbs: np array (N,H,W,3)
    @depths: (N,H,W)
    @masks: (N,H,W)
    @normal_maps: (N,H,W,3)
    @poses: (N,4,4)
    '''
    depths[depths<0.1] = BAD_DEPTH
    if masks is not None:
        rgbs[masks==0] = BAD_COLOR
        depths[masks==0] = BAD_DEPTH
        if normal_maps is not None:
            normal_maps[...,[1,2]] *= -1     # To OpenGL
            normal_maps[masks==0] = 0
        masks = masks[...,None]

    rgbs = (rgbs / 255.0).astype(np.float32)
    depths *= sc_factor
    depths = depths[...,None]
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor
    return rgbs,depths,masks,normal_maps,poses


class NeRFSmall(nn.Module):
    def __init__(self,num_layers=3,hidden_dim=64,geo_feat_dim=15,num_layers_color=4,hidden_dim_color=64,input_ch=3, input_ch_views=3):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
            if l!=num_layers-1:
                sigma_net.append(nn.ReLU(inplace=True))

        self.sigma_net = nn.Sequential(*sigma_net)
        torch.nn.init.constant_(self.sigma_net[-1].bias, 0.1)     # Encourage last layer predict positive SDF

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=True))
            if l!=num_layers_color-1:
                color_net.append(nn.ReLU(inplace=True))

        self.color_net = nn.Sequential(*color_net)

    def forward_sdf(self,x):
        '''
        @x: embedded positions
        '''
        h = self.sigma_net(x)
        sigma, geo_feat = h[..., 0], h[..., 1:]
        return sigma


    def forward(self, x):
        x = x.float()
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        h = self.sigma_net(h)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        color = self.color_net(h)

        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs


def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples



def get_camera_rays_np(H, W, K):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0,2])/K[0,0], -(j - K[1,2])/K[1,1], -np.ones_like(i)], axis=-1)
    return dirs



def get_masks(z_vals, target_d, truncation, cfg, dir_norm=None):
    valid_depth_mask = (target_d>=cfg['near']*cfg['sc_factor']) & (target_d<=cfg['far']*cfg['sc_factor'])
    front_mask = (z_vals < target_d - truncation)
    back_mask = (z_vals > target_d + truncation*cfg['neg_trunc_ratio'])

    sdf_mask = (1.0 - front_mask.float()) * (1.0 - back_mask.float()) * valid_depth_mask

    num_fs_samples = front_mask.sum()
    num_sdf_samples = sdf_mask.sum()
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 0.5
    sdf_weight = 1.0 - fs_weight
    return front_mask.bool(), sdf_mask.bool(), fs_weight, sdf_weight


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, cfg, return_mask=False, sample_weights=None, rays_d=None):
    dir_norm = rays_d.norm(dim=-1,keepdim=True)
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation, cfg, dir_norm=dir_norm)
    front_mask = front_mask.bool()

    mask = (target_d>cfg['far']*cfg['sc_factor']) & (predicted_sdf<cfg['fs_sdf'])

    fs_loss = torch.mean(((predicted_sdf-cfg['fs_sdf']) * mask)**2 * sample_weights) * fs_weight

    mask = front_mask & (target_d<=cfg['far']*cfg['sc_factor']) & (predicted_sdf<1)
    empty_loss = torch.mean(torch.abs(predicted_sdf-1) * mask * sample_weights)
    sdf_loss = torch.mean(((z_vals + predicted_sdf * truncation) * sdf_mask - target_d * sdf_mask)**2 * sample_weights) * sdf_weight

    if return_mask:
        return fs_loss,sdf_loss,empty_loss, front_mask,sdf_mask
    return fs_loss, sdf_loss, empty_loss



def ray_box_intersection_batch(origins, dirs, bounds):
    '''
    @origins: (N,3) origin and directions. In the same coordinate frame as the bounding box
    @bounds: (2,3) xyz_min and max
    '''
    if not torch.is_tensor(origins):
        origins = torch.tensor(origins)
        dirs = torch.tensor(dirs)
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds)

    dirs = dirs/(torch.norm(dirs,dim=-1,keepdim=True)+1e-10)
    inv_dirs = 1/dirs
    bounds = bounds[None].expand(len(dirs),-1,-1)   #(N,2,3)

    sign = torch.zeros((len(dirs),3)).long().to(dirs.device)  #(N,3)
    sign[:,0] = (inv_dirs[:,0] < 0)
    sign[:,1] = (inv_dirs[:,1] < 0)
    sign[:,2] = (inv_dirs[:,2] < 0)

    tmin = (torch.gather(bounds[...,0],dim=1,index=sign[:,0].reshape(-1,1)).reshape(-1) - origins[:,0]) * inv_dirs[:,0]   #(N)
    tmin[tmin<0] = 0
    tmax = (torch.gather(bounds[...,0],dim=1,index=1-sign[:,0].reshape(-1,1)).reshape(-1) - origins[:,0]) * inv_dirs[:,0]
    tymin = (torch.gather(bounds[...,1],dim=1,index=sign[:,1].reshape(-1,1)).reshape(-1) - origins[:,1]) * inv_dirs[:,1]
    tymin[tymin<0] = 0
    tymax = (torch.gather(bounds[...,1],dim=1,index=1-sign[:,1].reshape(-1,1)).reshape(-1) - origins[:,1]) * inv_dirs[:,1]

    ishit = torch.ones(len(dirs)).bool().to(dirs.device)
    ishit[(tmin > tymax) | (tymin > tmax)] = 0
    tmin[tymin>tmin] = tymin[tymin>tmin]
    tmax[tymax<tmax] = tymax[tymax<tmax]

    tzmin = (torch.gather(bounds[...,2],dim=1,index=sign[:,2].reshape(-1,1)).reshape(-1) - origins[:,2]) * inv_dirs[:,2]
    tzmin[tzmin<0] = 0
    tzmax = (torch.gather(bounds[...,2],dim=1,index=1-sign[:,2].reshape(-1,1)).reshape(-1) - origins[:,2]) * inv_dirs[:,2]

    ishit[(tmin > tzmax) | (tzmin > tmax)] = 0
    tmin[tzmin>tmin] = tzmin[tzmin>tmin]   #(N)
    tmax[tzmax<tmax] = tzmax[tzmax<tmax]

    tmin[ishit==0] = -1
    tmax[ishit==0] = -1

    return tmin, tmax
