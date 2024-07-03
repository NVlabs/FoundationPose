/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */



#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <stdint.h>
#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>
#include <set>
#include "common.h"
#include "Eigen/Dense"





/**
 * @brief
 *
 * @tparam scalar_t
 * @param z_sampled
 * @param z_in_out
 * @param z_vals
 * @return __global__
 */
template <typename scalar_t>
__global__ void sample_rays_uniform_occupied_voxels_kernel(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_sampled, const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> z_in_out, torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_vals)
{
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  const int i_sample = blockIdx.y * blockDim.y + threadIdx.y;
  if (i_ray>=z_sampled.size(0)) return;
  if (i_sample>=z_sampled.size(1)) return;

  int i_box = 0;
  float z_remain = z_sampled[i_ray][i_sample];
  auto z_in_out_cur_ray = z_in_out[i_ray];
  const float eps = 1e-4;
  const int max_n_box = z_in_out.size(1);

  if (z_in_out_cur_ray[0][0]==0) return;

  while (1)
  {
    if (i_box>=max_n_box)
    {
      if (z_remain<=eps)
      {
        z_vals[i_ray][i_sample] = z_in_out_cur_ray[max_n_box-1][1];
      }
      else
      {
        printf("ERROR sample_rays_uniform_occupied_voxels_kernel: z_remain=%f, i_ray=%d, i_sample=%d, i_box=%d, z_in_out_cur_ray=(%f,%f)\n",z_remain,i_ray,i_sample,i_box,z_in_out_cur_ray[i_box][0],z_in_out_cur_ray[i_box][1]);
        for (int i=0;i<z_in_out.size(1);i++)
        {
          printf("z_in_out_cur_ray[%d]=(%f,%f)\n",i,z_in_out_cur_ray[i][0],z_in_out_cur_ray[i][1]);
        }
        while (1){};
      }


      return;
    }

    if (z_in_out_cur_ray[i_box][0]==0)
    {
      if (z_remain<=eps && i_box>=1)
      {
        z_vals[i_ray][i_sample] = z_in_out_cur_ray[i_box-1][1];
        return;
      }
      else
      {
        printf("ERROR sample_rays_uniform_occupied_voxels_kernel: z_remain=%f, i_ray=%d, i_sample=%d, i_box=%d, z_in_out_cur_ray=(%f,%f)\n",z_remain,i_ray,i_sample,i_box,z_in_out_cur_ray[i_box][0],z_in_out_cur_ray[i_box][1]);
        for (int i=0;i<z_in_out.size(1);i++)
        {
          printf("z_in_out_cur_ray[%d]=(%f,%f)\n",i,z_in_out_cur_ray[i][0],z_in_out_cur_ray[i][1]);
        }
        while (1){};
      }
    }

    float box_len = z_in_out_cur_ray[i_box][1]-z_in_out_cur_ray[i_box][0];
    if (z_remain<=box_len)
    {
      z_vals[i_ray][i_sample] = z_in_out_cur_ray[i_box][0] + z_remain;
      return;
    }
    z_remain -= box_len;
    i_box++;
  }
}

at::Tensor sampleRaysUniformOccupiedVoxels(const at::Tensor z_in_out,  const at::Tensor z_sampled, at::Tensor z_vals)
{
  CHECK_INPUT(z_in_out);
  CHECK_INPUT(z_sampled);
  CHECK_INPUT(z_vals);
  AT_ASSERTM(z_vals.sizes()==z_sampled.sizes());

  const int N_rays = z_sampled.sizes()[0];
  const int N_samples = z_sampled.sizes()[1];
  const int threadx = 32;
  const int thready = 32;

  AT_DISPATCH_FLOATING_TYPES(z_in_out.type(), "sample_rays_uniform_occupied_voxels_kernel", ([&]
  {
    sample_rays_uniform_occupied_voxels_kernel<scalar_t><<<{divCeil(N_rays,threadx),divCeil(N_samples,thready)}, {threadx,thready}>>>(z_sampled.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),z_in_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),z_vals.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return z_vals;
}

template<class scalar_t>
__global__ void postprocessOctreeRayTracingKernel(const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ray_index, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> depth_in_out, const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> unique_intersect_ray_ids, const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> start_poss, torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depths_in_out_padded)
{
  const int unique_id_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (unique_id_pos>=unique_intersect_ray_ids.size(0)) return;
  const int i_ray = unique_intersect_ray_ids[unique_id_pos];

  int i_intersect = 0;
  auto cur_depths_in_out_padded = depths_in_out_padded[i_ray];
  for (int i=start_poss[unique_id_pos];i<ray_index.size(0);i++)
  {
    if (ray_index[i]!=i_ray) break;
    if (depth_in_out[i][0]==0 || depth_in_out[i][1]==0) break;
    if (depth_in_out[i][0]>depth_in_out[i][1]) continue;
    if (abs(depth_in_out[i][1]-depth_in_out[i][0])<1e-4) continue;

    cur_depths_in_out_padded[i_intersect][0] = depth_in_out[i][0];
    cur_depths_in_out_padded[i_intersect][1] = depth_in_out[i][1];

    i_intersect++;

  }
}

at::Tensor postprocessOctreeRayTracing(const at::Tensor ray_index, const at::Tensor depth_in_out, const at::Tensor unique_intersect_ray_ids, const at::Tensor start_poss, const int max_intersections, const int N_rays)
{
  CHECK_INPUT(ray_index);
  CHECK_INPUT(depth_in_out);
  CHECK_INPUT(start_poss);

  const int n_unique_ids = unique_intersect_ray_ids.sizes()[0];
  at::Tensor depths_in_out_padded = at::zeros({N_rays,max_intersections,2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false));
  dim3 threads = {256};
  dim3 blocks = {divCeil(n_unique_ids,threads.x)};
  AT_DISPATCH_FLOATING_TYPES(depth_in_out.type(), "postprocessOctreeRayTracingKernel", ([&]
  {
    postprocessOctreeRayTracingKernel<scalar_t><<<blocks,threads>>>(ray_index.packed_accessor32<long,1,torch::RestrictPtrTraits>(), depth_in_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), unique_intersect_ray_ids.packed_accessor32<long,1,torch::RestrictPtrTraits>(), start_poss.packed_accessor32<long,1,torch::RestrictPtrTraits>(), depths_in_out_padded.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  return depths_in_out_padded;
}


at::Tensor calculateBarycentricCoordinate3D(const at::Tensor &triangle, const at::Tensor &p)
{
  auto vector_A = triangle[1]-triangle[2];
  auto vector_B = triangle[1]-triangle[0];
  auto normal = vector_A.cross(vector_B);
  auto areaABC = (normal * at::cross(triangle[1]-triangle[0], triangle[2]-triangle[0])).sum();
  auto areaPBC = (normal * at::cross(triangle[1]-p, triangle[2]-p)).sum();
  auto areaPCA = (normal * at::cross(triangle[2]-p, triangle[0]-p)).sum();
  at::Tensor ws = at::zeros({3}).to(torch::kFloat32);
  ws[0] = areaPBC / areaABC;
  ws[1] = areaPCA / areaABC;
  ws[2] = 1-ws[0]-ws[1];
  return ws;
}


__device__ Eigen::Vector3f calculateBarycentricCoordinate3DKernel(const Eigen::Matrix<float,3,3> &triangle, const Eigen::Vector3f &p)
{
  Eigen::Vector3f vector_A = triangle.row(1)-triangle.row(2);
  Eigen::Vector3f vector_B = triangle.row(1)-triangle.row(0);
  Eigen::Vector3f normal = vector_A.cross(vector_B);
  float areaABC = (normal.array() * (triangle.row(1)-triangle.row(0)).cross(triangle.row(2)-triangle.row(0)).transpose().array()).sum();
  float areaPBC = (normal.array() * (triangle.row(1).transpose()-p).cross(triangle.row(2).transpose()-p).array()).sum();
  float areaPCA = (normal.array() * (triangle.row(2).transpose()-p).cross(triangle.row(0).transpose()-p).array()).sum();
  Eigen::Vector3f w;
  w[0] = areaPBC / areaABC;
  w[1] = areaPCA / areaABC;
  w[2] = 1-w[0]-w[1];
  return w;

}


__device__ void calculateBarycentricCoordinate2DKernel(const Eigen::Matrix<float,3,2> &triangle, const Eigen::Vector2f &p, Eigen::Vector3f &w)
{
  Eigen::Vector2f CA = triangle.row(0)-triangle.row(2);
  Eigen::Vector2f AC = -CA;
  Eigen::Vector2f CP = p-triangle.row(2).transpose();
  Eigen::Vector2f AB = triangle.row(1)-triangle.row(0);
  Eigen::Vector2f AP = p-triangle.row(0).transpose();
  Eigen::Matrix2f numerator, denominator;
  numerator << CA, CP;
  denominator<<AB, AC;
  w(1) = numerator.determinant()/denominator.determinant();
  numerator << AB, AP;
  w(2) = numerator.determinant()/denominator.determinant();
  w(0) = 1-w(1)-w(2);
}



template<class scalar_t>
__global__ void rayColorToTextureImageKernel(const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> F, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> V, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> hit_locations, const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> hit_face_ids, const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> uvs_tex, torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> uvs)
{
  const int i_hit = blockIdx.x*blockDim.x + threadIdx.x;
  if (i_hit>=hit_locations.size(0)) return;

  auto face = F[hit_face_ids[i_hit]];
  Eigen::Matrix3f tri_v = Eigen::Matrix3f::Zero();
  for (int r=0;r<3;r++)
  {
    for (int c=0;c<3;c++)
    {
      tri_v(r,c) = V[face[r]][c];
    }
  }
  auto location = hit_locations[i_hit];
  Eigen::Vector3f p(location[0], location[1], location[2]);
  auto w = calculateBarycentricCoordinate3DKernel(tri_v, p);

  Eigen::Matrix<float,3,2> tri_uvs;
  for (int i=0;i<3;i++)
  {
    for (int j=0;j<2;j++)
    {
      tri_uvs(i,j) = uvs_tex[face[i]][j];
    }
  }

  Eigen::Vector2f cur_uv = tri_uvs.transpose() * w;
  uvs[i_hit][0] = cur_uv(0);
  uvs[i_hit][1] = cur_uv(1);
}



void rayColorToTextureImageCUDA(const at::Tensor &F, const at::Tensor &V, const at::Tensor &hit_locations, const at::Tensor &hit_face_ids, const at::Tensor &uvs_tex, at::Tensor &uvs)
{
  CHECK_CONTIGUOUS(F);
  CHECK_CONTIGUOUS(V);
  CHECK_CONTIGUOUS(hit_locations);
  CHECK_CONTIGUOUS(hit_face_ids);
  CHECK_CONTIGUOUS(uvs_tex);

  dim3 threads = {512};
  dim3 blocks = {divCeil(int(hit_locations.sizes()[0]),threads.x)};

  AT_DISPATCH_FLOATING_TYPES(V.type(), "rayColorToTextureImageKernel", ([&]
  {
    rayColorToTextureImageKernel<scalar_t><<<blocks,threads>>>(F.packed_accessor32<long,2,torch::RestrictPtrTraits>(), V.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), hit_locations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), hit_face_ids.packed_accessor32<long,1,torch::RestrictPtrTraits>(), uvs_tex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(), uvs.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));
}



__device__ bool isLineIntersectLine(Eigen::Vector2f l1p1, Eigen::Vector2f l1p2, Eigen::Vector2f l2p1, Eigen::Vector2f l2p2)
{
  float q = (l1p1(1) - l2p1(1)) * (l2p2(0) - l2p1(0)) - (l1p1(0) - l2p1(0)) * (l2p2(1) - l2p1(1));
  float d = (l1p2(0) - l1p1(0)) * (l2p2(1) - l2p1(1)) - (l1p2(1) - l1p1(1)) * (l2p2(0) - l2p1(0));

  if ( d == 0 )
  {
    return false;
  }

  float r = q / d;

  q = (l1p1(1) - l2p1(1)) * (l1p2(0) - l1p1(0)) - (l1p1(0) - l2p1(0)) * (l1p2(1) - l1p1(1));
  float s = q / d;

  if( r < 0 || r > 1 || s < 0 || s > 1 )
  {
    return false;
  }

  return true;
}

__device__ bool isLineIntersectSquare(const Eigen::Vector2f &la, const Eigen::Vector2f &lb, const Eigen::Vector2f &p0, const Eigen::Vector2f &p1, const Eigen::Vector2f &p2, const Eigen::Vector2f &p3)
{
  return isLineIntersectLine(la, lb, p0, p1) || isLineIntersectLine(la, lb, p1, p2) || isLineIntersectLine(la, lb, p2, p3) || isLineIntersectLine(la, lb, p3, p0);
}


__device__ bool isPixelInsideTriangle(const Eigen::Matrix<float,3,2> &triangle, Eigen::Vector2f p, Eigen::Vector3f &w)
{
  calculateBarycentricCoordinate2DKernel(triangle, p, w);
  for (int j=0;j<3;j++)
  {
    if (w(j)<0) return false;
  }
  return true;
}
