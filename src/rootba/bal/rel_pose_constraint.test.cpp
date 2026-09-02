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
#include "rootba/bal/rel_pose_constraint.hpp"

#include "rootba/testing/test_jacobian.hpp"
#include "rootba/testing/test_types.hpp"

namespace rootba {

#if defined(ROOTBA_INSTANTIATIONS_FLOAT) || \
    defined(ROOTBA_INSTANTIATIONS_DOUBLE)

template <typename Scalar_>
class RelPoseConstraintTest : public ::testing::Test {
 public:
  using Scalar = Scalar_;
};

TYPED_TEST_SUITE(RelPoseConstraintTest, ScalarTestTypes);

TYPED_TEST(RelPoseConstraintTest, ErrorIsZeroForExactMeasurement) {
  using Scalar = typename TestFixture::Scalar;
  using SO3 = Sophus::SO3<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;
  using Vec3 = Vec<Scalar, 3>;

  const Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision() * 2;

  const SE3 T_a_w(SO3::exp(Vec3::Random()), Vec3::Random());
  const SE3 T_b_w(SO3::exp(Vec3::Random()), Vec3::Random());

  const SE3 T_a_b = T_a_w * T_b_w.inverse();

  const Vec<Scalar, 6> res = relative_pose_error(T_a_b, T_a_w, T_b_w);

  EXPECT_TRUE(res.isZero(prec)) << "res: " << res.transpose();
}

TYPED_TEST(RelPoseConstraintTest, Jacobians) {
  using Scalar = typename TestFixture::Scalar;
  using SO3 = Sophus::SO3<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;
  using Vec3 = Vec<Scalar, 3>;
  using Vec6 = Vec<Scalar, 6>;
  using Mat6 = Mat<Scalar, 6, 6>;

  const SE3 T_a_w(SO3::exp(Vec3::Random()), Vec3::Random());
  const SE3 T_b_w(SO3::exp(Vec3::Random()), Vec3::Random());

  // measurement that is close to, but not exactly, the current estimate
  const SE3 T_a_b = T_a_w * T_b_w.inverse() *
                    SE3(SO3::exp(Vec3::Random() / 10), Vec3::Random() / 10);

  Mat6 d_res_d_xi_a;
  Mat6 d_res_d_xi_b;
  relative_pose_error(T_a_b, T_a_w, T_b_w, &d_res_d_xi_a, &d_res_d_xi_b);

  const Vec6 x0 = Vec6::Zero();

  test_jacobian(
      "d_res_d_xi_a", d_res_d_xi_a,
      [&](const Vec6& x) {
        // same left perturbation as BalProblem::Keyframe::apply_inc_pose
        const SE3 T_a_w_new = Sophus::se3_expd(x) * T_a_w;
        return relative_pose_error(T_a_b, T_a_w_new, T_b_w);
      },
      x0);

  test_jacobian(
      "d_res_d_xi_b", d_res_d_xi_b,
      [&](const Vec6& x) {
        const SE3 T_b_w_new = Sophus::se3_expd(x) * T_b_w;
        return relative_pose_error(T_a_b, T_a_w, T_b_w_new);
      },
      x0);
}

#endif

}  // namespace rootba
