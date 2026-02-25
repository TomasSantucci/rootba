## Intro

This project has been adapted to preserve the stereo constraints of a multicamera setup.

Only square root optimization is implemented.

To support this, the Keyframe structure was added to represent the set of frames corresponding to the same timestamp. It contains the IMU pose (T_w_i). Using the Calibration object, attribute of `BalProblem`, one can access the transformation from the IMU to the corresponding cameras (T_c_i). Since the problem state contains only keyframe poses, we must compute errors, residuals, and Jacobians with respect to the IMU pose.

Additionally, intrinsics are no longer optimized and must be provided using a calibration file with the `--calibration-file` argument.

Because we now optimize IMU poses, the structure of a landmark block has been modified. Previously, each camera could have only one observation for a landmark, meaning that in a given landmark block, the column of a camera had only two non-zero rows. Now, since the columns of J_p represent IMU poses, each column can have up to 2*n non-zero rows, where n is the number of cameras in the setup.

Another conceptual difference is that Jacobians were previously computed with respect to the camera pose (d_res_d_cam), but are now computed with respect to the IMU pose (d_res_d_imu). The (possibly incorrect) derivation of this Jacobian can be found in the `linearize_landmark` method of `landmark_block_base.ipp`.

Note that since we no longer optimize intrinsics and use a fixed T_c_i, we cannot normalize the problem, which requires us to use the `--no-normalize` argument.

## How to run

Compile with:

```bash
./scripts/build-rootba.sh
```

A list of pre-exported maps is available in `data/stereo`, along with the calibration files used to generate them in `data/calib`. You can run it with:

```bash
./bin/bal_gui --input data/stereo/MOO01.json --solver-type SQUARE_ROOT --calibration-file data/calib/msdmo_calib.json --no-normalize
```

## Problems to check

### 1

The linear model does not appear to correctly represent the real model. This can be observed in nearly every dataset. The l_diff indicates that the cost should decrease, but f_diff shows the opposite. When both decrease, l_diff is not sufficiently similar to f_diff. This suggests a potential issue with the linearization. This leads to many rejected iterations and causes lambda to increase to very large values. For example:

```bash
	[EVAL] f_diff -4.8196e+05 l_diff 5.5251e+05 step_quality -8.7231e-01 ri1 5.6967e+05 ri2 1.0516e+06
```

### 2

In some cases, such as with `MOO13.json`, the sum of the weighted error decreases while the mean residual error increases. This should not occur, especially when the number of observations remains constant before and after optimization.

```bash
Iteration 0, error: 5.6967e+05 (mean res: 1.82, num: 33353), error valid: 5.6967e+05 (mean res: 1.82, num: 33353)
Final Cost: error: 3.9156e+05 (mean res: 2.36, num: 33353), error valid: 3.9156e+05 (mean res: 2.36, num: 33353)
```
