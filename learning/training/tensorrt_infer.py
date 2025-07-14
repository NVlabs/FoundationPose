#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import torch
import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import List
import time

# If you face the following issue:
#  "pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
#  Add "import pycuda.autoinit", this is needed to initialize cuda!
# import pycuda.autoinit


TRT_DYNAMIC_DIM = -1

class HostDeviceMem(object):
    """Simple helper data class to store Host and Device memory."""

    def __init__(self, host_mem, device_mem, binding):
        self.host = host_mem
        self.device = device_mem
        self.binding = binding

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device) + "\nBinding:\n" + str(self.binding)

    def __repr__(self):
        return self.__str__()




class TensorRTInfer:
    def __init__(self, engine=None, engine_path: str = None, batch_size: int = 1, verbose: bool = False):
        """
        Initialize TensorRT inference engine.

        Args:
            engine_path (str): Path to the TensorRT engine file
            batch_size (int): Batch size for inference
        """
        self.batch_size = batch_size
        self.engine_path = engine_path
        self.verbose = verbose

        if engine is not None:
            self.engine = engine
        else:
            # Load engine
            assert engine_path is not None, "engine_path is required when engine is not provided"
            with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

    def infer(self, input_tensors_list, glctx=None):
        """
        Safe TensorRT integration with existing nvdiffrast context for multiple inputs/outputs
        
        Args:
            input_tensors_list: List of PyTorch tensors for input
            glctx: nvdiffrast RasterizeCudaContext (already created)
        
        Returns:
            Dictionary mapping output tensor names to PyTorch tensors
        """

        # Ensure PyTorch CUDA context is established
        if not input_tensors_list[0].is_cuda:
            input_tensors_list = [tensor.cuda() for tensor in input_tensors_list]

        
        # Process all input and output tensors
        output_tensors_dict = {
            'trans': [],
            'rot': []
        }

        # Debug: Print engine information
        if self.verbose:
            print(f"Engine has {self.engine.num_io_tensors} I/O tensors")
        
        input_tensor_names = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            if self.verbose:
                print(f"Tensor {i}: name={tensor_name}, mode={tensor_mode}, shape={tensor_shape}, dtype={tensor_dtype}")
            
            if tensor_mode == trt.TensorIOMode.INPUT:
                input_tensor_names.append(tensor_name)
        
        if self.verbose:
            print(f"Input tensor names: {input_tensor_names}")
            print(f"Number of input tensors provided: {len(input_tensors_list)}")
        
        # Set input tensors
        for i, tensor_name in enumerate(input_tensor_names):
            if i < len(input_tensors_list):
                input_tensor = input_tensors_list[i]
                if self.verbose:
                    print(f"Setting input {i}: {tensor_name} with shape {input_tensor.shape}")
                binding_shape = tuple(input_tensor.shape)
                binding_shape = tuple([self.batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in binding_shape])
                self.context.set_input_shape(tensor_name, tuple(input_tensor.shape))
                self.context.set_tensor_address(tensor_name, input_tensor.data_ptr())

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            if tensor_mode == trt.TensorIOMode.OUTPUT:
                # Handle output tensors
                output_shape = tuple(self.context.get_tensor_shape(tensor_name))
                if self.verbose:
                    print(f"Creating output tensor: {tensor_name} with shape {output_shape}")
                
                # Determine output dtype based on engine
                engine_dtype = self.engine.get_tensor_dtype(tensor_name)
                if engine_dtype == trt.DataType.FLOAT:
                    output_dtype = torch.float32
                elif engine_dtype == trt.DataType.HALF:
                    output_dtype = torch.float16
                else:
                    output_dtype = torch.float32  # Default fallback
                
                output_tensor = torch.empty(
                    output_shape, 
                    dtype=output_dtype, 
                    device=input_tensors_list[0].device
                )
                if tensor_name == "output1":
                    output_tensors_dict['trans'] = output_tensor
                elif tensor_name == "output2":
                    output_tensors_dict['rot'] = output_tensor
                self.context.set_tensor_address(tensor_name, output_tensor.data_ptr())
        
        # Verify all binding shapes are specified
        assert self.context.all_binding_shapes_specified, "Not all input shapes are specified"
        
        # Execute inference using current stream (shared with nvdiffrast)
        current_stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=current_stream.cuda_stream)
        
        return output_tensors_dict


    def __del__(self):
        """Cleanup CUDA resources"""
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'engine'):
            del self.engine

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    engine_paths = [
        f"{code_dir}/../../weights/2023-10-28-18-33-37/model_best.plan",
        f"{code_dir}/../../weights/2024-01-11-20-02-45/model_best.plan"
    ]

    max_batch_size = 252
    num_inferences_per_engine = 10
    
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

    total_inference_time = [0.0 for _ in range(len(engine_paths))]
    for j, engine_path in enumerate(engine_paths):
        trt_infer = TensorRTInfer(engine_path=engine_path, batch_size=max_batch_size, verbose=False)
        for i in range(num_inferences_per_engine):
            batch_size = np.random.randint(1, max_batch_size + 1)
            input_tensor1 = torch.randn((batch_size, 160, 160, 6), dtype=torch.float32).to("cuda")
            input_tensor2 = torch.randn((batch_size, 160, 160, 6), dtype=torch.float32).to("cuda")

            start_time = time.time()

            output = trt_infer.infer([input_tensor1, input_tensor2])
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_inference_time[j] += inference_time
            print(f"Running inference {i+1}/{num_inferences_per_engine} | Inference time: {inference_time:.2f} ms")

    print("-"*100)
    for j in range(len(engine_paths)):
        print(f"Average inference time for Model {j}: {total_inference_time[j] / num_inferences_per_engine:.2f} ms")


