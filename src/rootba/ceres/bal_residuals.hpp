/**
BSD 3-Clause License

This file is part of the RootBA project.
https://github.com/NikolausDemmel/rootba

Copyright (c) 2021, Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#pragma once

#include <basalt/camera/generic_camera.hpp>
#include <ceres/dynamic_autodiff_cost_function.h>

#include "rootba/ceres/types.hpp"
#include "rootba/util/eigen_types.hpp"

namespace rootba {

// Generic-camera Ceres residual with dynamic intrinsics size.
template <int Options = 0>
class BalGenericReprojectionError {
 public:
  BalGenericReprojectionError(const Vec2d& obs,
                              const basalt::GenericCamera<double>& cam_model,
                              int intrinsics_size)
      : obs_(obs), cam_model_(cam_model), intrinsics_size_(intrinsics_size) {}

  // DynamicAutoDiffCostFunction expects this signature.
  template <class T>
  bool operator()(const T* const* parameters, T* residual) const {
    return (*this)(parameters[0], parameters[1], residual);
  }

  template <class T>
  bool operator()(const T* camera, const T* landmark, T* residual) const {
    Eigen::Map<const Sophus::SE3<T>> T_c_w(camera);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(landmark);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> res_vec(residual);

    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> intrinsics(
        camera + 7, intrinsics_size_);

    auto cam = cam_model_.template cast<T>();
    cam.setParams(intrinsics);  // set intrinsics from the parameter block

    auto p_cam = T_c_w * p_w;
    if constexpr (Options & VALID_PROJECTIONS_ONLY) {
      if (p_cam.z() < Sophus::Constants<T>::epsilonSqrt()) {
        res_vec.setZero();
        return true;
      }
    }

    Eigen::Matrix<T, 4, 1> p_cam_h = p_cam.homogeneous();
    Eigen::Matrix<T, 2, 1> p_proj;
    cam.project(p_cam_h, p_proj);
    res_vec = p_proj - obs_.cast<T>();
    return true;
  }

 private:
  Vec2d obs_;
  basalt::GenericCamera<double> cam_model_;
  int intrinsics_size_;
};

template <int Options = 0>
inline ceres::CostFunction* create_bal_reprojection_cost(
    const Vec2d& obs, const basalt::GenericCamera<double>& cam_model,
    int intrinsics_size) {
  using Functor = BalGenericReprojectionError<Options>;
  auto* functor = new Functor(obs, cam_model, intrinsics_size);
  auto* cost = new ceres::DynamicAutoDiffCostFunction<Functor>(functor);
  cost->AddParameterBlock(7 + intrinsics_size);
  cost->AddParameterBlock(3);
  cost->SetNumResiduals(2);
  return cost;
}

}  // namespace rootba
