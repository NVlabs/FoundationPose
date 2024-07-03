/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once

#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


inline int divCeil(int a, int b)
{
  return (a+b-1)/b;
};


at::Tensor sampleRaysUniformOccupiedVoxels(const at::Tensor z_in_out,  const at::Tensor z_sampled, at::Tensor z_vals);
at::Tensor postprocessOctreeRayTracing(const at::Tensor ray_index, const at::Tensor depth_in_out, const at::Tensor unique_intersect_ray_ids, const at::Tensor start_poss, const int max_intersections, const int N_rays);
void rayColorToTextureImageCUDA(const at::Tensor &F, const at::Tensor &V, const at::Tensor &hit_locations, const at::Tensor &hit_face_ids, const at::Tensor &uvs_tex, at::Tensor &uvs);