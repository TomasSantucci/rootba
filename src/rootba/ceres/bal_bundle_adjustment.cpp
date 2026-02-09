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

#include "rootba/ceres/bal_bundle_adjustment.hpp"

#include <ceres/ceres.h>

#include "rootba/bal/bal_pipeline_summary.hpp"
#include "rootba/ceres/ba_log_utils.hpp"
#include "rootba/ceres/bal_iteration_callback.hpp"
#include "rootba/ceres/bal_residuals.hpp"
#include "rootba/ceres/option_utils.hpp"
#include "rootba/util/tbb_utils.hpp"

namespace rootba {

void bundle_adjust_ceres(BalProblem<double>& bal_problem,
                         const SolverOptions& solver_options, BaLog* log) {
  // Limit tbb threads (the effecitve limit is also passed to ceres in its
  // options).
  ScopedTbbThreadLimit thread_limit(solver_options.num_threads);

  // options
  const bool projection_validitity_check =
      ceres_use_projection_validity_check(solver_options);

  // prepare camera state with per-camera dynamic sizes
  std::vector<VecXd> camera_blocks;
  camera_blocks.reserve(bal_problem.num_cameras());
  for (const auto& cam : bal_problem.cameras()) {
    camera_blocks.push_back(cam.params());
  }

  // manually define ordering for SC: lms 0, cams 1 (for large problems it can
  // sometimes be suboptimal if determined automatically)
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();

  // setup camera parameter blocks
  ceres::Problem problem;
  std::vector<double*> camera_ptrs;
  camera_ptrs.reserve(bal_problem.num_cameras());
  for (size_t i = 0; i < camera_blocks.size(); ++i) {
    VecXd& cam_block = camera_blocks[i];
    camera_ptrs.push_back(cam_block.data());

    const int intrinsics_size = bal_problem.cameras()[i].intrinsics.getN();
    const int block_size = 7 + intrinsics_size;
    const int euclid_size = 3 + intrinsics_size;  // translation + intrinsics

    auto* manifold =
        new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                   ceres::EuclideanManifold<ceres::DYNAMIC>>(
            ceres::EigenQuaternionManifold(),
            ceres::EuclideanManifold<ceres::DYNAMIC>(euclid_size));

    problem.AddParameterBlock(camera_ptrs.back(), block_size, manifold);
    ordering->AddElementToGroup(camera_ptrs.back(), 1);
  }

  // setup landmark parameter blocks and residuals
  ceres::LossFunction* loss_function =
      ceres_create_loss_function(solver_options.residual);
  for (auto& lm : bal_problem.landmarks()) {
    problem.AddParameterBlock(lm.p_w.data(), 3);
    ordering->AddElementToGroup(lm.p_w.data(), 0);

    for (const auto& [cam_idx, obs] : lm.obs) {
      const int intrinsics_size =
          bal_problem.cameras()[cam_idx].intrinsics.getN();
      const auto& cam_model = bal_problem.cameras()[cam_idx].intrinsics;
      ceres::CostFunction* cost =
          projection_validitity_check
              ? create_bal_reprojection_cost<VALID_PROJECTIONS_ONLY>(
                    obs.pos, cam_model, intrinsics_size)
              : create_bal_reprojection_cost<>(obs.pos, cam_model,
                                               intrinsics_size);

      problem.AddResidualBlock(cost, loss_function, camera_ptrs[cam_idx],
                               lm.p_w.data());
    }
  }

  // prepare options
  ceres::Solver::Options options;
  options.linear_solver_ordering = ordering;
  set_ceres_options(options, solver_options);

  // set iteration callback if logging
  std::unique_ptr<BalIterationCallback> callback;
  if (log != nullptr) {
    callback = std::make_unique<BalIterationCallback>(
        *log, bal_problem, camera_blocks, solver_options);
    options.callbacks.push_back(callback.get());
    options.update_state_every_iteration = true;
  }

  // run solver
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  print_ceres_summary(summary, solver_options);

  // update camera state
  // write back optimized camera blocks into BalProblem
  for (size_t i = 0; i < bal_problem.cameras().size(); ++i) {
    bal_problem.cameras()[i].from_params(camera_blocks[i]);
  }

  // log summary if logging
  if (log != nullptr) {
    log_ceres_summary(*log, "bal_ceres", summary);
    log->static_data.timing.optimize = summary.total_time_in_seconds;
    log->static_data.timing.update_total();
  }
}

}  // namespace rootba
