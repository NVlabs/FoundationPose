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



namespace Utils
{


// Difference angle in radian
float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2)
{
  float cos = ((R1 * R2.transpose()).trace()-1) / 2.0;
  cos = std::max(std::min(cos, 1.0f), -1.0f);
  return std::acos(cos);
}


} // namespace Utils
