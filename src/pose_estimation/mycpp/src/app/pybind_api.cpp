/*
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include "Utils.h"
#include <boost/algorithm/string.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;



//@angle_diff: unit is degree
//@dist_diff: unit is meter
vectorMatrix4f cluster_poses(float angle_diff, float dist_diff, const vectorMatrix4f &poses_in, const vectorMatrix4f &symmetry_tfs)
{
  printf("num original candidates = %d\n",poses_in.size());
  vectorMatrix4f poses_out;
  poses_out.push_back(poses_in[0]);

  const float radian_thres = angle_diff/180.0*M_PI;

  for (int i=1;i<poses_in.size();i++)
  {
    bool isnew = true;
    Eigen::Matrix4f cur_pose = poses_in[i];
    for (const auto &cluster:poses_out)
    {
      Eigen::Vector3f t0 = cluster.block(0,3,3,1);
      Eigen::Vector3f t1 = cur_pose.block(0,3,3,1);

      if ((t0-t1).norm()>=dist_diff)
      {
        continue;
      }

      for (const auto &tf: symmetry_tfs)
      {
        Eigen::Matrix4f cur_pose_tmp = cur_pose*tf;
        float rot_diff = Utils::rotationGeodesicDistance(cur_pose_tmp.block(0,0,3,3), cluster.block(0,0,3,3));
        if (rot_diff < radian_thres)
        {
          isnew = false;
          break;
        }
      }

      if (!isnew) break;
    }

    if (isnew)
    {
      poses_out.push_back(poses_in[i]);
    }
  }

  printf("num of pose after clustering: %d\n",poses_out.size());
  return poses_out;
}





PYBIND11_MODULE(mycpp, m)
{
  m.def("cluster_poses", &cluster_poses, py::call_guard<py::gil_scoped_release>());
}