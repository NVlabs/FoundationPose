#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict
import pickle
import sys
import trimesh
import numpy as np
import cv2
import torch as th
import einops
import nvdiffrast.torch as dr
from icecream import ic

from pkm.util.torch_util import dcn
from pkm.real.util import draw_pose_axes
from pkm.util.math_util import invert_transform
from pkm.real.vos.cutie_vos import CutieVOS
from pkm.util.img_util import _chw, _hwc


def _apply_extrinsics(x):
    return x


def guess_diameter(frame, q: float = 0.95):
    center = frame['cloud'][..., :3].mean(
        dim=-2,
        keepdim=True
    )
    radius = th.linalg.norm(
        frame['cloud'][..., :3] - center,
        dim=-1)
    # radius = radius.mean(dim=-1)
    radius = th.quantile(radius,
                         q,
                         dim=-1)
    return 2.0 * radius


class _FP:
    @dataclass
    class Config:
        mesh_file: str = '/tmp/docker/down_pig/merged/merged.obj'
        fp_dir: str = '/input/FoundationPose/'
        debug_dir: str = ''
        debug: int = 0

        reg_iter: int = 5
        track_iter: int = 2

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Extra FP-related imports
        sys.path.append(cfg.fp_dir)
        from run_demo import (
            ScorePredictor,
            PoseRefinePredictor,
            FoundationPose
        )
        sys.path.pop(-1)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()  # needed?
        self.scorer = scorer
        self.refiner = refiner
        self.glctx = glctx
        self.fp_cls = FoundationPose
        self.fp = None

        self.mesh = trimesh.load(cfg.mesh_file,
                                 force='mesh')

    @property
    def cami(self):
        return self.__cami_last

    def setup(self, frame: Dict[str, th.Tensor]):
        cfg = self.cfg

        mesh = self.mesh
        if mesh is None:
            # Get inputs from the frame.
            center = frame['cloud'][..., 0:3].mean(
                dim=-2, keepdim=True)
            p = frame['cloud'][..., 0:3] - center
            n = frame['cloud'][..., 6:9]
        else:
            # From mesh.
            p = mesh.vertices
            n = mesh.vertex_normals

        self.fp = self.fp_cls(
            model_pts=p,
            model_normals=n,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=None,
            debug=cfg.debug,
            glctx=self.glctx
        )

        if mesh is None:
            self.fp.pose_last = th.eye(4)[None]
            self.fp.pose_last[..., :3, 3] = center
            self.fp.diameter = guess_diameter(frame)
        else:
            if ('label' in frame) and (frame['label'] is not None):
                # Initialize from 'best view'
                # i = np.random.randint(4)
                i = np.argmax(
                    dcn(einops.reduce(frame['label'], 'n ... -> n', 'sum'))
                )
            else:
                # Workaround for now...
                i = 0

            ic(
                frame['intrinsics'][i].shape,
                frame['color'][i].shape,
                frame['depth'][i].shape,
                frame['label'][i].shape,
            )

            pose = self.fp.register(
                dcn(frame['intrinsics'][i]),
                dcn(frame['color'][i]),
                dcn(frame['depth'][i]),
                # ONLY used once at the beginning
                dcn(frame['label'][i]),
                iteration=cfg.reg_iter
            )

            self.__pose_last = self.fp.pose_last
            self.__cami_last = i
        return pose

    def __call__(self, frame: Dict[str, th.Tensor]):
        cfg = self.cfg

        # Track from 'best view'
        if True:
            # i = np.random.randint(4)
            i = th.argmax(
                einops.reduce(frame['label'], 'n ... -> n', 'sum')
            )
        else:
            i = 0
        extra = {}

        # pose_last = _apply_extrinsics(
        #     self.__pose_last,
        #     frame['extrinsics'][self.__cami_last],
        #     frame['extrinsics'][i]
        # )

        frame = dict(frame)
        frame['extrinsics'] = th.as_tensor(frame['extrinsics'],
                                           dtype=th.float32,
                                           device='cuda')
        pose_last = (
            invert_transform(frame['extrinsics'][i])
            @ frame['extrinsics'][self.__cami_last]
            @ self.__pose_last
        )
        pose = self.fp.track_one(rgb=dcn(frame['color'][i]),
                                 depth=dcn(frame['depth'][i]),
                                 K=dcn(frame['intrinsics'][i]),
                                 iteration=cfg.track_iter,
                                 extra=extra,
                                 pose_last=pose_last)
        self.__pose_last = extra['pose']
        self.__cami_last = i
        return pose


def load_data(device: str = 'cuda'):
    from pkm.real.seg_v2 import SegmenterApp
    with open('/tmp/docker/pig/5.pkl', 'rb') as fp:
        data = pickle.load(fp)

    dev_ids = ['233622074125', '233622074736', '233622070987', '101622072564']
    vos = CutieVOS(CutieVOS.Config(
        cfg_file='/home/user/Cutie/cutie/config/gui_config.yaml',
        ckpt='/home/user/Cutie/weights/cutie-base-mega.pth',
        amp=True), device)

    # Compatibility...
    if isinstance(data, dict):
        if isinstance(data['intrinsics'], dict):
            data['intrinsics'] = np.stack(
                [data['intrinsics'][dev_id] for dev_id in dev_ids],
                axis=-3
            )

        if isinstance(data['extrinsics'], dict):
            data['extrinsics'] = np.stack(
                [data['extrinsics'][dev_id] for dev_id in dev_ids],
                axis=-3
            )

    T: int = len(data['color'])
    if len(data['extrinsics'].shape) == 3:
        data['extrinsics'] = einops.repeat(data['extrinsics'],
                                           '... -> T ...',
                                           T=T)

    if len(data['intrinsics'].shape) == 3:
        data['intrinsics'] = einops.repeat(data['intrinsics'],
                                           '... -> T ...',
                                           T=T)

    for i in range(T):
        ks = 'extrinsics,intrinsics,color,depth'.split(',')
        data_i = {k: data[k][i] for k in ks}
        if (i == 0) and ('label' not in data):
            if False:
                # Generate labels online
                labels = []
                for j in range(len(data['color'][i])):
                    cv2.imwrite(F'/tmp/color_{j:02d}.png',
                                dcn(data['color'][i][j]))
                    SegmenterApp.run(SegmenterApp.Config(
                        ckpt_file='/tmp/docker/repvit_sam.pt',
                        sam_dir='/home/user/Grounded-Segment-Anything/'
                    ),
                        F'/tmp/color_{j:02d}.png',
                        F'/tmp/label_{j:02d}.png')
                    label = cv2.imread(
                        F'/tmp/label_{j:02d}.png',
                        cv2.IMREAD_UNCHANGED)
                    labels.append(label != 0)
                labels = np.stack(labels, axis=0)
            else:
                # Load initial labels from filesystem
                labels = [
                    cv2.imread(F'/tmp/label_{j:02d}.png', cv2.IMREAD_UNCHANGED)
                    for j in range(len(data['color'][i]))]
                labels = np.stack(labels, axis=0)
            data_i['label'] = labels

            color_22 = einops.rearrange(data_i['color'],
                                        '(nh nw) h w c -> c (nh h) (nw w)',
                                        nh=2, nw=2)
            label_22 = einops.rearrange(labels,
                                        '(nh nw) h w -> (nh h) (nw w)',
                                        nh=2, nw=2)
            if vos is not None:
                inputs = (th.as_tensor(color_22 / 255.0,
                                       dtype=th.float32,
                                       device=device),
                          th.as_tensor(label_22[None],
                                       dtype=th.float32,
                                       device=device))
                print([xx.shape for xx in inputs])
                _ = vos(*inputs)
        elif 'label' in data:
            # load cached labels
            data_i['label'] = data['label'][i]
        elif vos is not None:
            # Add labels, via VOS
            color_22 = einops.rearrange(data_i['color'],
                                        '(nh nw) h w c -> c (nh h) (nw w)',
                                        nh=2, nw=2)
            label_22 = vos(th.as_tensor(color_22 / 255.0,
                                        dtype=th.float32,
                                        device=device))
            label = einops.rearrange(label_22,
                                     '(nh h) (nw w) -> (nh nw) h w',
                                     nh=2, nw=2)
            data_i['label'] = label
        yield data_i

    print(data.keys())
    # => extrinsics, intrinsics,
    # => color, depth, joint_pos
    return


def main():
    # load_data()
    # return

    fp = _FP(_FP.Config())

    setup: bool = True
    count = 0
    for data in load_data():
        if setup:
            pose = fp.setup(data)
            setup = False
            # continue
        else:
            pose = fp(data)
        # i = 0
        i = fp.cami
        vis = draw_pose_axes(dcn(data['color'][i]),
                             dcn(pose),
                             dcn(data['intrinsics'][i])
                             )
        cv2.imwrite(F'/tmp/docker/vis-{count:02d}.png', vis)
        count += 1
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
