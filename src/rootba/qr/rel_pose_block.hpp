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

#include <mutex>

#include <Eigen/Dense>

#include "rootba/bal/bal_problem.hpp"
#include "rootba/bal/rel_pose_constraint.hpp"
#include "rootba/cg/block_sparse_matrix.hpp"
#include "rootba/util/assert.hpp"
#include "rootba/util/cast.hpp"

namespace rootba {

// Linearized relative pose constraint between two keyframes.
//
// This is the analogue of the IMU / marginalization prior blocks in Basalt's
// square root VIO: the residual doesn't depend on any landmark, so there is
// nothing to marginalize and the (dense, 6 x 12) jacobian is appended
// unchanged below the Q2' Jp rows of the reduced camera system. All operations
// therefore mirror those of `LandmarkBlock`, with Q2' Jp == Jp and Q2' r == r.
template <typename Scalar_, int POSE_SIZE_ = 6>
class RelPoseBlock {
 public:
  using Scalar = Scalar_;
  static constexpr int POSE_SIZE = POSE_SIZE_;
  static constexpr int RES_SIZE = 6;

  using Vec6 = Vec<Scalar, 6>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Mat6 = Mat<Scalar, 6, 6>;
  using MatRC = Mat<Scalar, RES_SIZE, 2 * POSE_SIZE>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  using Keyframes = typename BalProblem<Scalar>::Keyframes;

  static_assert(POSE_SIZE == 6, "relative pose blocks assume 6-dof poses");

  explicit RelPoseBlock(const RelPoseConstraint<Scalar>& constraint)
      : constraint_(&constraint) {
    Jp_.setZero();
    r_.setZero();
  }

  size_t frame_a() const { return unsigned_cast(constraint_->frame_a); }
  size_t frame_b() const { return unsigned_cast(constraint_->frame_b); }

  size_t col_a() const { return frame_a() * POSE_SIZE; }
  size_t col_b() const { return frame_b() * POSE_SIZE; }

  static constexpr size_t num_rows() { return RES_SIZE; }

  // Linearize at the current keyframe poses. Returns the (non-robustified)
  // error contribution 0.5 * ||r||^2.
  Scalar linearize(const Keyframes& keyframes) {
    const auto& kf_a = keyframes.at(frame_a());
    const auto& kf_b = keyframes.at(frame_b());

    Mat6 d_res_d_xi_a;
    Mat6 d_res_d_xi_b;
    const Vec6 res =
        relative_pose_error(constraint_->T_a_b, kf_a.T_i_w, kf_b.T_i_w,
                            &d_res_d_xi_a, &d_res_d_xi_b);

    const Mat6& sqrt_info = constraint_->sqrt_info;

    Jp_.template leftCols<POSE_SIZE>() = sqrt_info * d_res_d_xi_a;
    Jp_.template rightCols<POSE_SIZE>() = sqrt_info * d_res_d_xi_b;
    r_ = sqrt_info * res;

    return Scalar(0.5) * r_.squaredNorm();
  }

  bool is_numerical_failure() const {
    return !Jp_.array().isFinite().all() || !r_.array().isFinite().all();
  }

  void scale_Jp_cols(const VecX& jacobian_scaling) {
    Jp_.template leftCols<POSE_SIZE>() *=
        jacobian_scaling.template segment<POSE_SIZE>(col_a()).asDiagonal();
    Jp_.template rightCols<POSE_SIZE>() *=
        jacobian_scaling.template segment<POSE_SIZE>(col_b()).asDiagonal();
  }

  void add_Jp_diag2(VecX& res) const {
    res.template segment<POSE_SIZE>(col_a()) +=
        Jp_.template leftCols<POSE_SIZE>().colwise().squaredNorm();
    res.template segment<POSE_SIZE>(col_b()) +=
        Jp_.template rightCols<POSE_SIZE>().colwise().squaredNorm();
  }

  void add_Jp_T_Jp_blockdiag(BlockDiagonalAccumulator<Scalar>& accu) const {
    const auto Ja = Jp_.template leftCols<POSE_SIZE>();
    const auto Jb = Jp_.template rightCols<POSE_SIZE>();
    accu.add(frame_a(), MatX(Ja.transpose() * Ja));
    accu.add(frame_b(), MatX(Jb.transpose() * Jb));
  }

  // res += Jp' * Jp * x
  void add_Jp_T_Jp_mult_x(VecX& res, const VecX& x,
                          std::vector<std::mutex>* pose_mutex = nullptr) const {
    const Vec6 Jx = get_Jp_postmult_x(x);
    add_Jp_premult_x(res, Jx, pose_mutex);
  }

  // res += Jp' * r
  void add_Jp_T_r(VecX& res) const { add_Jp_premult_x(res, r_); }

  // Jp * x, restricted to this block's two pose columns
  Vec6 get_Jp_postmult_x(const VecX& x) const {
    return Jp_.template leftCols<POSE_SIZE>() *
               x.template segment<POSE_SIZE>(col_a()) +
           Jp_.template rightCols<POSE_SIZE>() *
               x.template segment<POSE_SIZE>(col_b());
  }

  // res += Jp' * x_r
  void add_Jp_premult_x(VecX& res, const Vec6& x_r,
                        std::vector<std::mutex>* pose_mutex = nullptr) const {
    const Vec6 inc_a = Jp_.template leftCols<POSE_SIZE>().transpose() * x_r;
    const Vec6 inc_b = Jp_.template rightCols<POSE_SIZE>().transpose() * x_r;

    if (pose_mutex == nullptr) {
      res.template segment<POSE_SIZE>(col_a()) += inc_a;
      res.template segment<POSE_SIZE>(col_b()) += inc_b;
    } else {
      {
        std::scoped_lock lock(pose_mutex->at(frame_a()));
        res.template segment<POSE_SIZE>(col_a()) += inc_a;
      }
      {
        std::scoped_lock lock(pose_mutex->at(frame_b()));
        res.template segment<POSE_SIZE>(col_b()) += inc_b;
      }
    }
  }

  void add_triplets_Jp(size_t start_idx,
                       std::vector<Eigen::Triplet<Scalar>>& triplets) const {
    for (int row = 0; row < RES_SIZE; ++row) {
      for (int col = 0; col < POSE_SIZE; ++col) {
        triplets.emplace_back(start_idx + row, col_a() + col, Jp_(row, col));
        triplets.emplace_back(start_idx + row, col_b() + col,
                              Jp_(row, POSE_SIZE + col));
      }
    }
  }

  // model cost change, see LandmarkBlock::back_substitute:
  //   l_diff = -(J inc)' * (r + 0.5 * (J inc))
  void back_substitute(const VecX& pose_inc, Scalar& l_diff) const {
    const Vec6 Jinc = get_Jp_postmult_x(pose_inc);
    l_diff -= Jinc.transpose() * (Scalar(0.5) * Jinc + r_);
  }

  const Vec6& get_r() const { return r_; }
  const MatRC& get_Jp() const { return Jp_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // [d_res_d_xi_a | d_res_d_xi_b], pre-multiplied with the sqrt information
  MatRC Jp_;
  Vec6 r_;

  const RelPoseConstraint<Scalar>* constraint_;
};

}  // namespace rootba
