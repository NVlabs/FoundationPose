
import os,sys
import time
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import numpy as np
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet
from Utils import *

class PoseRefinePredictor:
  def __init__(self, runtime='torch', run_name='2023-10-28-18-33-37'):
    self.run_name = run_name
    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')
    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
    logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
        ckpt = ckpt['model']
        self.model.load_state_dict(ckpt)
        self.model.cuda().eval()

if __name__ == "__main__":
    predictor = PoseRefinePredictor(runtime='torch')
    num_inferences_per_engine = 1000
    for i in range(num_inferences_per_engine):
        A = torch.randn(1, 6, 160, 160).cuda()
        B = torch.randn(1, 6, 160, 160).cuda()
        start_time = time.time()
        output = predictor.model.forward(A, B)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Inference {i+1}/{num_inferences_per_engine} | Inference time: {inference_time:.2f} ms")