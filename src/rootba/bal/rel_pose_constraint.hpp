/**
BSD 3-Clause License

This file is part of the RootBA project.
https://github.com/NikolausDemmel/rootba

Copyright (c) 2021-2023, Nikolaus Demmel.
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

#include <basalt/utils/sophus_utils.hpp>

#include "rootba/bal/common_types.hpp"
#include "rootba/util/eigen_types.hpp"

namespace rootba {

// Relative pose error between two keyframes.
//
// Keyframes store the world-to-body pose `T_i_w` and are updated with a left
// perturbation `T_i_w <- expd(xi) * T_i_w` (see `BalProblem::Keyframe`), so
// both jacobians below are w.r.t. that same 6-dof parametrization, ordered as
// [translation; rotation] like `se3_logd` / `se3_expd`.
//
// With the estimated relative pose `T_a_b_est = T_a_w * T_b_w^-1` the residual
// is
//
//     res = logd( T_a_b_meas * T_a_b_est^-1 ),
//
// i.e. it vanishes when the estimate matches the measurement. The jacobians
// follow from
//
//     d/d xi_a: T_a_b_est^-1 -> T_a_b_est^-1 * expd(-xi_a)
//     d/d xi_b: T_a_b_est^-1 -> expd(xi_b) * T_a_b_est^-1
//
// where the second one is moved to the right of the residual with the
// (decoupled) adjoint of `T_a_b_est`.
template <typename Scalar>
Vec<Scalar, 6> relative_pose_error(const Sophus::SE3<Scalar>& T_a_b_meas,
                                   const Sophus::SE3<Scalar>& T_a_w,
                                   const Sophus::SE3<Scalar>& T_b_w,
                                   Mat<Scalar, 6, 6>* d_res_d_xi_a = nullptr,
                                   Mat<Scalar, 6, 6>* d_res_d_xi_b = nullptr) {
  using Mat3 = Mat<Scalar, 3, 3>;
  using Mat6 = Mat<Scalar, 6, 6>;
  using SO3 = Sophus::SO3<Scalar>;

  const Sophus::SE3<Scalar> T_a_b_est = T_a_w * T_b_w.inverse();
  const Vec<Scalar, 6> res = Sophus::se3_logd(T_a_b_meas * T_a_b_est.inverse());

  if (d_res_d_xi_a || d_res_d_xi_b) {
    Mat6 J;
    Sophus::rightJacobianInvSE3Decoupled(res, J);

    if (d_res_d_xi_a) {
      *d_res_d_xi_a = -J;
    }

    if (d_res_d_xi_b) {
      // adjoint of T_a_b_est for the decoupled parametrization, i.e. the map
      // eta with T_b_a * expd(xi) * T_a_b = expd(eta) to first order
      const Mat3 R_a_b = T_a_b_est.so3().matrix();
      Mat6 adj = Mat6::Zero();
      adj.template topLeftCorner<3, 3>() = R_a_b;
      adj.template bottomRightCorner<3, 3>() = R_a_b;
      adj.template topRightCorner<3, 3>() =
          SO3::hat(T_a_b_est.translation()) * R_a_b;

      *d_res_d_xi_b = J * adj;
    }
  }

  return res;
}

// Relative pose measurement between two keyframes of a `BalProblem`. Unlike
// reprojection residuals these don't involve any landmark, so in the square
// root solver they are appended directly to the reduced camera system (no
// nullspace projection needed); see `RelPoseBlock`.
template <typename Scalar>
struct RelPoseConstraint {
  using SE3 = Sophus::SE3<Scalar>;
  using Vec6 = Vec<Scalar, 6>;
  using Mat6 = Mat<Scalar, 6, 6>;

  // indices into `BalProblem::keyframes()`
  FrameIdx frame_a = INVALID_FRAME_IDX;
  FrameIdx frame_b = INVALID_FRAME_IDX;

  // measured pose of keyframe b's body frame expressed in keyframe a's
  SE3 T_a_b;

  // square root information matrix; the weighted residual is `sqrt_info * res`
  Mat6 sqrt_info = Mat6::Identity();

  // set `sqrt_info` from independent translation / rotation standard
  // deviations (in map units and radians respectively)
  void set_sigmas(Scalar sigma_translation, Scalar sigma_rotation) {
    Vec6 diag;
    diag.template head<3>().setConstant(Scalar(1) / sigma_translation);
    diag.template tail<3>().setConstant(Scalar(1) / sigma_rotation);
    sqrt_info = diag.asDiagonal();
  }

  template <typename Scalar2>
  RelPoseConstraint<Scalar2> cast() const {
    RelPoseConstraint<Scalar2> res;
    res.frame_a = frame_a;
    res.frame_b = frame_b;
    res.T_a_b = T_a_b.template cast<Scalar2>();
    res.sqrt_info = sqrt_info.template cast<Scalar2>();
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace rootba
