# Pose graph format

The solver can optimize keyframe poses and landmarks against a pose graph in
addition to the reprojection error. The pose graph is passed as a separate JSON
file:

```
bal --input map.json --calibration-file calib.json --pose-graph-file posegraph.json
```

Its edges become relative pose constraints between keyframes. The topology is
arbitrary — sequential odometry, covisibility and loop closure edges can be
mixed freely, and keyframes may appear in any number of edges (or in none).

## File layout

```json
{
  "edges": [
    {
      "kf_id_a": 1403636579763555584,
      "kf_id_b": 1403636580113555456,
      "T_a_b": [0.99937, -0.00768, -0.03454, -0.00038, -0.16366, 0.00213, 0.05019],
      "sigmas": [0.02, 0.02, 0.02, 0.005, 0.005, 0.005]
    },
    {
      "kf_id_a": 1403636579763555584,
      "kf_id_b": 1403636612313555456,
      "T_a_b": [0.99012, 0.01204, 0.13901, 0.00337, 1.20412, -0.03318, 0.41003],
      "information": [ ... 36 values, row major ... ]
    }
  ]
}
```

## Fields

### `kf_id_a`, `kf_id_b` (required)

The two keyframes the edge connects, using **the same ids as the input map**
(for the basalt map format that is the keyframe timestamp in nanoseconds, i.e.
the `id` field of the `keyframes` entries).

Edges referencing an id that is not in the map, and edges connecting a keyframe
to itself, are skipped with a warning. This is not an error, so a pose graph
covering a superset of the map still loads.

### `T_a_b` (optional)

The measured pose of keyframe `b` expressed in the frame of keyframe `a`, as
`[qw, qx, qy, qz, tx, ty, tz]`.

This is the **body (IMU) frame**, and the same layout and direction as `T_w_i`
in the map file, so

```
T_a_b == T_a_w * T_w_b == inverse(T_w_a) * T_w_b
```

Watch the direction: `T_a_b` maps points from `b`'s frame into `a`'s frame. If
your SLAM system stores `T_b_a`, invert it before writing.

If `T_a_b` is **omitted**, the relative pose is measured from the keyframe poses
of the input map. The edge then means "preserve the input geometry along this
connection", which is useful when your SLAM system emits a topology (e.g. a
covisibility graph) without its own relative pose estimates.

### Uncertainty (optional)

The residual is a 6-vector ordered `[translation; rotation]` (`se3_logd`
convention: translation first, then the `SO(3)` log). The weighted residual is
`sqrt_info * residual`, and the cost contribution is
`0.5 * ||sqrt_info * residual||^2`.

The first of these that is present on an edge wins:

| Field         | Size | Meaning                                                        |
|---------------|------|----------------------------------------------------------------|
| `sqrt_info`   | 36   | Row-major square root `S` of the information matrix, `S' S = Ω` |
| `information` | 36   | Row-major information matrix `Ω` (inverse covariance); its Cholesky factor is used |
| `sigmas`      | 6    | Independent standard deviations `[tx, ty, tz, rx, ry, rz]`      |

`information` must be positive definite. Translations are in the metric units of
the input dataset, rotations in radians.

Edges with none of these fall back to the command line defaults:

```
--pose-graph-sigma-translation 0.05   # metres, by default
--pose-graph-sigma-rotation    0.01   # radians
```

## Notes

- The pose graph is loaded **before** `--normalize` rescales the map. Sigmas are
  therefore always in the metric units of the input dataset, and normalization
  rescales the measurements along with the map so the cost is unchanged.
- `--min-obs-per-kf` removes keyframes after the pose graph is loaded. Edges
  touching a removed keyframe are dropped (the count is logged); they are not
  composed across the gap, so the graph can end up with holes.
- Pose graph residuals are included in the reported cost and residual count, so
  the numbers in the iteration log and in `ba_log.json` cover both terms. There
  is currently no per-term breakdown.
