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
#include "rootba/qr/rel_pose_block.hpp"

#include "rootba/testing/test_types.hpp"

namespace rootba {

#if defined(ROOTBA_INSTANTIATIONS_FLOAT) || \
    defined(ROOTBA_INSTANTIATIONS_DOUBLE)

namespace {

constexpr int NUM_KEYFRAMES = 4;
constexpr int POSE_SIZE = 6;
constexpr int NUM_COLS = NUM_KEYFRAMES * POSE_SIZE;

// Frame indices deliberately not adjacent and not starting at 0, so that a
// wrong column offset would show up.
constexpr FrameIdx FRAME_A = 1;
constexpr FrameIdx FRAME_B = 3;

}  // namespace

template <typename Scalar_>
class RelPoseBlockTest : public ::testing::Test {
 public:
  using Scalar = Scalar_;
};

TYPED_TEST_SUITE(RelPoseBlockTest, ScalarTestTypes);

// The matrix-free operations must agree with the dense jacobian spread over the
// full set of pose columns.
TYPED_TEST(RelPoseBlockTest, MatrixFreeOpsMatchDenseJacobian) {
  using Scalar = typename TestFixture::Scalar;
  using SO3 = Sophus::SO3<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;
  using Vec3 = Vec<Scalar, 3>;
  using Vec6 = Vec<Scalar, 6>;
  using VecX = Vec<Scalar, Eigen::Dynamic>;
  using MatX = Mat<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Keyframe = typename BalProblem<Scalar>::Keyframe;

  const Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision() * 10;

  std::vector<Keyframe> keyframes(NUM_KEYFRAMES);
  for (auto& kf : keyframes) {
    kf.T_i_w = SE3(SO3::exp(Vec3::Random()), Vec3::Random());
  }

  RelPoseConstraint<Scalar> constraint;
  constraint.frame_a = FRAME_A;
  constraint.frame_b = FRAME_B;
  constraint.T_a_b = keyframes[FRAME_A].T_i_w *
                     keyframes[FRAME_B].T_i_w.inverse() *
                     SE3(SO3::exp(Vec3::Random() / 10), Vec3::Random() / 10);
  constraint.set_sigmas(Scalar(0.05), Scalar(0.01));

  RelPoseBlock<Scalar> block(constraint);
  block.linearize(keyframes);
  ASSERT_FALSE(block.is_numerical_failure());

  // scaling exercises the column offsets as well
  VecX jacobian_scaling = VecX::Random(NUM_COLS).cwiseAbs().array() + Scalar(1);
  block.scale_Jp_cols(jacobian_scaling);

  // dense reference: full 6 x NUM_COLS jacobian
  MatX J = MatX::Zero(RelPoseBlock<Scalar>::RES_SIZE, NUM_COLS);
  J.template block<6, POSE_SIZE>(0, FRAME_A * POSE_SIZE) =
      block.get_Jp().template leftCols<POSE_SIZE>();
  J.template block<6, POSE_SIZE>(0, FRAME_B * POSE_SIZE) =
      block.get_Jp().template rightCols<POSE_SIZE>();
  const Vec6 r = block.get_r();

  const VecX x = VecX::Random(NUM_COLS);

  // J * x
  EXPECT_TRUE(block.get_Jp_postmult_x(x).isApprox(J * x, prec));

  // J' * x_r
  {
    const Vec6 x_r = Vec6::Random();
    VecX res = VecX::Zero(NUM_COLS);
    block.add_Jp_premult_x(res, x_r);
    EXPECT_TRUE(res.isApprox(J.transpose() * x_r, prec));
  }

  // J' * J * x
  {
    VecX res = VecX::Zero(NUM_COLS);
    block.add_Jp_T_Jp_mult_x(res, x);
    EXPECT_TRUE(res.isApprox(J.transpose() * (J * x), prec));
  }

  // J' * r
  {
    VecX res = VecX::Zero(NUM_COLS);
    block.add_Jp_T_r(res);
    EXPECT_TRUE(res.isApprox(J.transpose() * r, prec));
  }

  // squared column norms
  {
    VecX res = VecX::Zero(NUM_COLS);
    block.add_Jp_diag2(res);
    EXPECT_TRUE(res.isApprox(J.colwise().squaredNorm().transpose(), prec));
  }

  // diagonal blocks of J' * J
  {
    BlockDiagonalAccumulator<Scalar> accu;
    block.add_Jp_T_Jp_blockdiag(accu);
    const MatX JTJ = J.transpose() * J;
    for (FrameIdx idx : {FRAME_A, FRAME_B}) {
      const auto it = accu.block_diagonal.find(std::make_pair(idx, idx));
      ASSERT_NE(it, accu.block_diagonal.end());
      EXPECT_TRUE(it->second.isApprox(
          JTJ.template block<POSE_SIZE, POSE_SIZE>(idx * POSE_SIZE,
                                                   idx * POSE_SIZE),
          prec));
    }
    EXPECT_EQ(accu.block_diagonal.size(), 2u);
  }

  // triplets of the sparse jacobian
  {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    const size_t row_offset = 12;
    block.add_triplets_Jp(row_offset, triplets);

    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> sm(
        row_offset + RelPoseBlock<Scalar>::RES_SIZE, NUM_COLS);
    sm.setFromTriplets(triplets.begin(), triplets.end());

    const MatX dense = MatX(sm).bottomRows(RelPoseBlock<Scalar>::RES_SIZE);
    EXPECT_TRUE(dense.isApprox(J, prec));
    EXPECT_TRUE(MatX(sm).topRows(row_offset).isZero(prec));
  }

  // model cost change: -(J inc)' * (r + 0.5 * J inc)
  {
    Scalar l_diff = 0;
    block.back_substitute(x, l_diff);
    const Vec6 Jx = J * x;
    const Scalar expected = -Jx.dot(Scalar(0.5) * Jx + r);
    EXPECT_NEAR(l_diff, expected, prec * std::max(Scalar(1), std::abs(expected)));
  }
}

#endif

}  // namespace rootba
