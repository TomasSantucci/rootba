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

#include "rootba/bal/bal_problem.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <utility>

#include <absl/container/flat_hash_set.h>
#include <cereal/archives/binary.hpp>
#include <glog/logging.h>

#include "rootba/bal/bal_dataset_options.hpp"
#include "rootba/bal/bal_pipeline_summary.hpp"
#include "rootba/bal/bal_problem_io.hpp"
#include "rootba/cg/block_sparse_matrix.hpp"
#include "rootba/util/format.hpp"
#include "rootba/util/stl_utils.hpp"
#include "rootba/util/time_utils.hpp"

namespace rootba {

namespace {  // helper

template <typename T>
void fscan_or_throw(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    throw std::runtime_error("");
  }
}

template <typename T, int N>
void fscan_or_throw(FILE* fptr, Eigen::Matrix<T, N, 1>& values) {
  for (int i = 0; i < values.size(); ++i) {
    fscan_or_throw(fptr, "%lf", values.data() + i);
  }
}

void readcommentline_or_throw(FILE* fptr) {
  char buffer[1000];
  bool comment_ok = false;
  while (fgets(buffer, 1000, fptr) != nullptr) {
    size_t len = strlen(buffer);

    if (len == 0) {
      throw std::runtime_error("empty line; expected comment...");
    }

    // first part of line, check # character
    if (!comment_ok) {
      if (buffer[0] == '#') {
        comment_ok = true;
      } else {
        throw std::runtime_error("non-comment line; expected comment...");
      }
    }

    // check if we reached eol
    if (buffer[len - 1] == '\n') {
      return;
    }
  }

  // fgets failed
  throw std::runtime_error("could not read comment line");
}

template <typename T, int N, class RandomEngine>
Eigen::Matrix<T, N, 1> perturbation(const T sigma, RandomEngine& eng) {
  std::normal_distribution<T> normal;
  Eigen::Matrix<T, N, 1> vec;
  vec.setZero();
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] += normal(eng) * sigma;
  }
  return vec;
}

template <typename T>
T median_destructive(std::vector<T>& data) {
  int n = data.size();
  auto mid_point = data.begin() + n / 2;
  std::nth_element(data.begin(), mid_point, data.end());
  return *mid_point;
}

BalDatasetOptions::DatasetType autodetect_input_type(const std::string& path) {
  using std::filesystem::is_directory;
  const std::string filename = std::filesystem::path(path).filename();

  if (ends_with(filename, ".cereal")) {
    return BalDatasetOptions::DatasetType::ROOTBA;
  } else if (std::string::npos != filename.find("bundle")) {
    return BalDatasetOptions::DatasetType::BUNDLER;
  } else if (is_directory(path) && is_directory(path + "/sparse")) {
    return BalDatasetOptions::DatasetType::COLMAP;
  } else {
    // default to BAL
    return BalDatasetOptions::DatasetType::BAL;
  }
}

class BalProblemSaver : public FileSaver<cereal::BinaryOutputArchive> {
 public:
  using Scalar = double;

  inline BalProblemSaver(std::string path,
                         const BalProblem<Scalar>& bal_problem)
      : FileSaver(BAL_PROBLEM_FILE_INFO, std::move(path)),
        bal_problem_(bal_problem) {}

 protected:
  inline bool save_impl(cereal::BinaryOutputArchive& archive) override {
    archive(bal_problem_);
    return true;
  }

  inline std::string format_summary() const override {
    return bal_problem_.stats_to_string();
  }

 private:
  const BalProblem<Scalar>& bal_problem_;
};

class BalProblemLoader : public FileLoader<cereal::BinaryInputArchive> {
 public:
  using Scalar = double;

  inline BalProblemLoader(std::string path, BalProblem<Scalar>& bal_problem)
      : FileLoader(BAL_PROBLEM_FILE_INFO, std::move(path)),
        bal_problem_(bal_problem) {}

 protected:
  inline bool load_impl() override {
    (*archive_)(bal_problem_);
    return true;
  }

  inline std::string format_summary() const override {
    return bal_problem_.stats_to_string();
  }

 private:
  BalProblem<Scalar>& bal_problem_;
};

}  // namespace

template <typename Scalar>
BalProblem<Scalar>::BalProblem(const std::string& path) {
  load_bal(path);
}

template <typename Scalar>
void BalProblem<Scalar>::load_bal(const std::string& path) {
  FILE* fptr = std::fopen(path.c_str(), "r");
  if (fptr == nullptr) {
    LOG(FATAL) << "Could not open '{}'"_format(path);
  };

  try {
    // parse header
    int num_cams;
    int num_lms;
    int num_obs;
    fscan_or_throw(fptr, "%d", &num_cams);
    fscan_or_throw(fptr, "%d", &num_lms);
    fscan_or_throw(fptr, "%d", &num_obs);
    CHECK_GT(num_cams, 0);
    CHECK_GT(num_lms, 0);

    // clear memory and re-allocate
    if (cameras_.capacity() > unsigned_cast(num_cams)) {
      decltype(cameras_)().swap(cameras_);
    }
    if (landmarks_.capacity() > unsigned_cast(num_lms)) {
      decltype(landmarks_)().swap(landmarks_);
    }
    cameras_.resize(num_cams);
    landmarks_.resize(num_lms);

    // parse observations
    for (int i = 0; i < num_obs; ++i) {
      int cam_idx;
      int lm_idx;
      fscan_or_throw(fptr, "%d", &cam_idx);
      fscan_or_throw(fptr, "%d", &lm_idx);
      CHECK_GE(cam_idx, 0);
      CHECK_LT(cam_idx, num_cams);
      CHECK_GE(lm_idx, 0);
      CHECK_LT(lm_idx, num_lms);

      auto [obs, inserted] = landmarks_.at(lm_idx).obs.try_emplace(cam_idx);
      CHECK(inserted) << "Invalid file '{}'"_format(path);
      Eigen::Matrix<double, 2, 1> posd;
      fscan_or_throw(fptr, posd);
      obs->second.pos = posd.cast<Scalar>();

      // For the camera frame we assume the positive z axis pointing
      // forward in view direction and in the image, y is poiting down, x to the
      // right. In the original BAL formulation, the camera points in negative z
      // axis, y is up in the image. Thus when loading the data, we invert the y
      // and z camera axes (y also in the image) in the perspective projection,
      // we don't have the "minus" like in the original Snavely model.

      // invert y axis
      obs->second.pos.y() = -obs->second.pos.y();
    }

    // invert y and z axis (same as rotation around x by 180; self-inverse)
    const SO3 axis_inversion = SO3(Vec3(1, -1, -1).asDiagonal());

    // parse camera parameters
    for (int i = 0; i < num_cams; ++i) {
      Vec9 params;
      Eigen::Matrix<double, 9, 1> paramsd;
      fscan_or_throw(fptr, paramsd);
      params = paramsd.cast<Scalar>();

      auto& cam = cameras_.at(i);
      cam.T_c_w.so3() = axis_inversion * SO3::exp(params.template head<3>());
      cam.T_c_w.translation() = axis_inversion * params.template segment<3>(3);

      cam.intrinsics = CameraModel::fromString("bal");
      cam.intrinsics.setParams(params.template head<3>());
    }

    // parse landmark parameters
    for (int i = 0; i < num_lms; ++i) {
      Eigen::Matrix<double, 3, 1> p_wd;
      fscan_or_throw(fptr, p_wd);
      landmarks_.at(i).p_w = p_wd.cast<Scalar>();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse '{}'"_format(path);
  }

  if (!quiet_) {
    LOG(INFO)
        << "Loaded BAL problem ({} cams, {} lms, {} obs) from '{}'"_format(
               num_cameras(), num_landmarks(), num_observations(), path);
  }

  // Current implementation uses int to compute state vector indices
  CHECK_LT(num_cameras(), std::numeric_limits<int>::max() / CAM_STATE_SIZE);

  std::fclose(fptr);
}

template <typename Scalar>
void BalProblem<Scalar>::load_bundler(const std::string& path) {
  FILE* fptr = std::fopen(path.c_str(), "r");
  if (fptr == nullptr) {
    LOG(FATAL) << "Could not open '{}'"_format(path);
  };

  try {
    // expect one comment line
    readcommentline_or_throw(fptr);
    // parse header
    int num_cams;
    int num_lms;
    fscan_or_throw(fptr, "%d", &num_cams);
    fscan_or_throw(fptr, "%d", &num_lms);
    CHECK_GT(num_cams, 0);
    CHECK_GT(num_lms, 0);

    // clear memory and re-allocate
    cameras_.clear();
    if (cameras_.capacity() > unsigned_cast(num_cams)) {
      decltype(cameras_)().swap(cameras_);
    }
    cameras_.reserve(num_cams);

    // invert y and z axis (same as rotation around x by 180; self-inverse)
    const SO3 axis_inversion = SO3(Vec3(1, -1, -1).asDiagonal());

    // not all cameras are initialized; so keep mapping from index in loaded
    // file to actual index
    std::unordered_map<int, int> cam_idx_mapping;

    // parse cameras
    for (int i = 0; i < num_cams; ++i) {
      Eigen::Matrix<double, 15, 1> paramsd;
      fscan_or_throw(fptr, paramsd);

      if (paramsd(0) == 0) {
        // focal length 0 --> assume uninitialzed camera
        continue;
      }

      Eigen::Matrix<Scalar, 15, 1> params = paramsd.cast<Scalar>();

      // remember where camera i is in the cameras_ vector
      cam_idx_mapping[i] = int(cameras_.size());

      // create camera object
      auto& cam = cameras_.emplace_back();
      cam.intrinsics = CameraModel::fromString("bal");
      cam.intrinsics.setParams(params.template head<3>());
      Eigen::Map<Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>> R(params.data() +
                                                                 3);
      cam.T_c_w.so3() = axis_inversion * SO3(R);
      cam.T_c_w.translation() = axis_inversion * params.template tail<3>();
    }

    if (landmarks_.capacity() > unsigned_cast(num_lms)) {
      decltype(landmarks_)().swap(landmarks_);
    }
    landmarks_.resize(num_lms);

    // parse landmarks and observation list
    for (int i = 0; i < num_lms; ++i) {
      auto& lm = landmarks_.at(i);

      // parse 3 vector for position
      Eigen::Matrix<double, 3, 1> p_wd;
      fscan_or_throw(fptr, p_wd);
      lm.p_w = p_wd.cast<Scalar>();

      // parse and ignore 3 vector for color
      fscan_or_throw(fptr, p_wd);

      // parse view list
      int num_obs;
      fscan_or_throw(fptr, "%d", &num_obs);

      for (int j = 0; j < num_obs; ++j) {
        int cam_idx;
        int feature_idx;  // we ignore this
        fscan_or_throw(fptr, "%d", &cam_idx);
        fscan_or_throw(fptr, "%d", &feature_idx);

        Eigen::Matrix<double, 2, 1> posd;
        fscan_or_throw(fptr, posd);

        const bool cam_exists = cam_idx_mapping.count(cam_idx);
        if (cam_exists) {
          auto [obs, inserted] =
              lm.obs.try_emplace(cam_idx_mapping.at(cam_idx));
          CHECK(inserted) << "Invalid file '{}'"_format(path);
          obs->second.pos = posd.cast<Scalar>();

          // For the camera frame we assume the positive z axis pointing
          // forward in view direction and in the image, y is poiting down, x to
          // the right. In the original BAL formulation, the camera points in
          // negative z axis, y is up in the image. Thus when loading the data,
          // we invert the y and z camera axes (y also in the image) in the
          // perspective projection, we don't have the "minus" like in the
          // original Snavely model.

          // invert y axis
          obs->second.pos.y() = -obs->second.pos.y();
        }
      }
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse '{}'"_format(path);
  }

  if (!quiet_) {
    LOG(INFO)
        << "Loaded BAL problem ({} cams, {} lms, {} obs) from '{}'"_format(
               num_cameras(), num_landmarks(), num_observations(), path);
  }

  // Current implementation uses int to compute state vector indices
  CHECK_LT(num_cameras(), std::numeric_limits<int>::max() / CAM_STATE_SIZE);

  std::fclose(fptr);
}

template <typename Scalar>
void BalProblem<Scalar>::load_colmap(const std::string& path_str) {
  using Quaternion = Eigen::Quaternion<Scalar>;
  using std::getline;
  using std::ifstream;
  using std::istringstream;
  using std::string;
  using std::unordered_map;
  using std::filesystem::is_directory;
  using std::filesystem::path;

  path dir = path{path_str} / "sparse" / "0";
  if (!is_directory(dir)) {
    LOG(FATAL) << "Invalid COLMAP dataset '{}'"_format(dir.string());
  }

  cameras_.clear();
  landmarks_.clear();

  try {
    unordered_map<ssize_t, size_t> pid_to_idx{};

    // Read cameras.txt
    /* Example format:
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 3
    1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
    2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
    3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531
    */
    struct ColmapCamera {
      std::string model;
      std::vector<Scalar> params;
      Scalar cx = 0;
      Scalar cy = 0;
    };
    unordered_map<ssize_t, ColmapCamera> colmap_cameras;

    path cameras_txt = dir / "cameras.txt";
    ifstream f(cameras_txt);
    string line;
    while (getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;

      istringstream ss(line);

      ssize_t camera_id = 0;
      string model;
      ssize_t width = 0;
      ssize_t height = 0;
      bool read = bool(ss >> camera_id >> model >> width >> height);
      CHECK(read) << "cameras.txt: '{}'"_format(line);
      camera_id -= 1;  // COLMAP camera IDs are 1-based

      if (model == "SIMPLE_RADIAL") {
        Scalar f = 0;
        Scalar cx = 0;
        Scalar cy = 0;
        Scalar k1 = 0;
        bool read = bool(ss >> f >> cx >> cy >> k1);
        CHECK(read) << "cameras.txt: '{}'"_format(line);
        colmap_cameras[camera_id] = ColmapCamera{"bal", {f, k1, 0}, cx, cy};
      } else if (model == "PINHOLE") {
        Scalar fx = 0, fy = 0, cx = 0, cy = 0;
        bool read = bool(ss >> fx >> fy >> cx >> cy);
        CHECK(read) << "cameras.txt: '{}'"_format(line);
        colmap_cameras[camera_id] =
            ColmapCamera{"pinhole", {fx, fy, 0, 0}, cx, cy};
      } else if (model == "SIMPLE_PINHOLE") {
        Scalar f = 0, cx = 0, cy = 0;
        bool read = bool(ss >> f >> cx >> cy);
        CHECK(read) << "cameras.txt: '{}'"_format(line);
        colmap_cameras[camera_id] =
            ColmapCamera{"pinhole", {f, f, 0, 0}, cx, cy};
      } else if (model == "FULL_OPENCV") {
        Scalar fx = 0, fy = 0, cx = 0, cy = 0, k1 = 0, k2 = 0, p1 = 0, p2 = 0,
               k3 = 0, k4 = 0, k5 = 0, k6 = 0;
        bool read = bool(ss >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >>
                         k3 >> k4 >> k5 >> k6);
        CHECK(read) << "cameras.txt: '{}'"_format(line);
        colmap_cameras[camera_id] =
            ColmapCamera{"pinhole-radtan8",
                         {fx, fy, 0, 0, k1, k2, p1, p2, k3, k4, k5, k6},
                         cx,
                         cy};
      } else if (model == "OPENCV_FISHEYE") {
        Scalar fx = 0, fy = 0, cx = 0, cy = 0, k1 = 0, k2 = 0, k3 = 0, k4 = 0;
        bool read = bool(ss >> fx >> fy >> cx >> cy >> k1 >> k2 >> k3 >> k4);
        CHECK(read) << "cameras.txt: '{}'"_format(line);
        colmap_cameras[camera_id] =
            ColmapCamera{"kb4", {fx, fy, 0, 0, k1, k2, k3, k4}, cx, cy};
      } else {
        LOG(FATAL) << "Not implemented: COLMAP camera model '{}'"_format(model);
      }
    }

    // Read images.txt
    /* Example format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 2, mean observations per image: 2
    1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1
    P1180141.JPG 2362.39 248.498 58396 1784.7 268.254 59027 1784.7 268.254 -1 2
    0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1
    P1180142.JPG 1190.83 663.957 23056 1258.77 640.354 59070
    */
    path images_txt = dir / "images.txt";
    f = ifstream{images_txt};
    while (getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;

      istringstream ss(line);

      ssize_t image_id = 0;
      Scalar qw = 0;
      Scalar qx = 0;
      Scalar qy = 0;
      Scalar qz = 0;
      Scalar tx = 0;
      Scalar ty = 0;
      Scalar tz = 0;
      ssize_t camera_id = 0;
      string image_name;
      bool read = bool(ss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >>
                       tz >> camera_id >> image_name);
      CHECK(read) << "images.txt: '{}'"_format(line);

      // Convert COLMAP id to zero-based index
      camera_id -= 1;
      image_id -= 1;
      if (cameras_.size() <= size_t(image_id)) cameras_.resize(image_id + 1);

      Camera& cam = cameras_.at(image_id);
      CHECK(colmap_cameras.find(camera_id) != colmap_cameras.end())
          << "missing camera_id=" << camera_id;
      ColmapCamera& colcam = colmap_cameras.at(camera_id);
      cam.T_c_w.so3() = SO3(Quaternion(qw, qx, qy, qz));
      cam.T_c_w.translation() = Vec3{tx, ty, tz};
      cam.intrinsics = CameraModel::fromString(colcam.model);
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> intr;
      intr.resize(colcam.params.size());
      for (size_t i = 0; i < colcam.params.size(); ++i) {
        intr(static_cast<int>(i)) = colcam.params[i];
      }
      cam.intrinsics.setParams(intr);

      getline(f, line);
      ss = istringstream(line);
      Scalar x = 0;
      Scalar y = 0;
      ssize_t pid = 0;
      while (ss >> x >> y >> pid) {
        if (pid <= 0) continue;

        size_t lmidx = -1;
        if (pid_to_idx.find(pid) == pid_to_idx.end()) {
          pid_to_idx[pid] = landmarks_.size();
          landmarks_.emplace_back();
          lmidx = landmarks_.size() - 1;
        } else {
          lmidx = pid_to_idx[pid];
        }

        // Note: colmap can have >1 obs of same point in one image, use last one
        landmarks_.at(lmidx).obs[image_id] = {{x - colcam.cx, y - colcam.cy}};
      }
    }

    // Read points3D.txt
    /* points3D.txt example format:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 3, mean track length: 3.3334
    63390 1.67241 0.292931 0.609726 115 121 122 1.33927 16 6542 15 7345 6 6714
    14 7227 63376 2.01848 0.108877 -0.0260841 102 209 250 1.73449 16 6519 15
    7322 14 7212 8 3991 63371 1.71102 0.28566 0.53475 245 251 249 0.612829 118
    4140 117 12
    */
    path points3d_txt = dir / "points3D.txt";
    f = ifstream{points3d_txt};
    while (getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;

      istringstream ss(line);
      ssize_t pid = 0;
      Scalar x = 0;
      Scalar y = 0;
      Scalar z = 0;
      Scalar r = 0;
      Scalar g = 0;
      Scalar b = 0;
      Scalar error = 0;
      bool read = bool(ss >> pid >> x >> y >> z >> r >> g >> b >> error);
      CHECK(read) << "points3D.txt: '{}'"_format(line);
      CHECK(pid_to_idx.find(pid) != pid_to_idx.end()) << "missing pid=" << pid;
      Landmark& lm = landmarks_.at(pid_to_idx.at(pid));
      lm.p_w = {x, y, z};
      lm.color = {uint8_t(r), uint8_t(g), uint8_t(b)};
    }

    // Read image_id_to_frame_id.txt which maps each colmap image ID to a basalt
    // TimeCamId
    path image_id_to_frame_id_txt = dir / "image_id_to_frame_id.txt";
    if (std::filesystem::exists(image_id_to_frame_id_txt)) {
      f = ifstream{image_id_to_frame_id_txt};
      while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream ss(line);
        ssize_t image_id = 0;
        ssize_t frame_id = 0;
        ssize_t cam_id = 0;
        bool read = bool(ss >> image_id >> frame_id >> cam_id);
        CHECK(read) << "image_id_to_frame_id.txt: '{}'"_format(line);
        sequential_colmap_id_to_basalt_id[image_id] = frame_id;
        image_id_to_cam_id[image_id] = cam_id;
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR)
        << "Exception caught while loading COLMAP dataset '{}'\n'{}'"_format(
               dir.string(), e.what());
  }
}

template <typename Scalar>
bool BalProblem<Scalar>::load_rootba(const std::string& path) {
  if constexpr (std::is_same_v<Scalar, double>) {
    return BalProblemLoader(path, *this).load();
  } else {
    BalProblem<double> temp;
    bool res = BalProblemLoader(path, temp).load();
    *this = temp.copy_cast<Scalar>();
    return res;
  }
}

template <typename Scalar>
bool BalProblem<Scalar>::save_rootba(const std::string& path) {
  if constexpr (std::is_same_v<Scalar, double>) {
    return BalProblemSaver(path, *this).save();
  } else {
    auto temp = copy_cast<double>();
    return BalProblemSaver(path, temp).save();
  }
}

template <typename Scalar>
bool BalProblem<Scalar>::save_bal(const std::string& path) {
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  FILE* fptr = std::fopen(path.c_str(), "w");
  if (fptr == nullptr) {
    LOG(FATAL) << "Could not open file for writing: '" << path << "'";
    return false;
  }

  int num_cams = static_cast<int>(cameras_.size());
  int num_lms = static_cast<int>(landmarks_.size());
  int num_obs = 0;
  for (const auto& lm : landmarks_) num_obs += lm.obs.size();

  fprintf(fptr, "%d %d %d\n", num_cams, num_lms, num_obs);

  for (int lm_idx = 0; lm_idx < int(landmarks_.size()); ++lm_idx) {
    const auto& lm = landmarks_[lm_idx];
    for (const auto& [cam_idx, obs] : lm.obs) {
      fprintf(fptr, "%d %d ", cam_idx, lm_idx);
      fprintf(fptr, "%lf %lf\n", obs.pos.x(), -obs.pos.y());  // Invert Y
    }
  }

  const SO3 axis_inversion = SO3(Vec3(1, -1, -1).asDiagonal());
  for (const auto& cam : cameras_) {
    SO3 R = axis_inversion * cam.T_c_w.so3();
    Vec3 r = R.log();
    Vec3 t = axis_inversion * cam.T_c_w.translation();
    Vec3 intr = cam.intrinsics.getParam();
    fprintf(fptr, "%lf %lf %lf ", r.x(), r.y(), r.z());
    fprintf(fptr, "%lf %lf %lf ", t.x(), t.y(), t.z());
    fprintf(fptr, "%lf %lf %lf\n", intr(0), intr(1), intr(2));
  }

  for (const auto& lm : landmarks_) {
    fprintf(fptr, "%lf %lf %lf\n", lm.p_w.x(), lm.p_w.y(), lm.p_w.z());
  }
  std::fclose(fptr);
  return true;
}

template <typename Scalar>
bool BalProblem<Scalar>::save_euroc(const std::string& path,
                                    const basalt::Calibration<double>& calib) {
  FILE* fptr = std::fopen(path.c_str(), "w");
  if (fptr == nullptr) {
    LOG(FATAL) << "Could not open file for writing: '" << path << "'";
    return false;
  }
  fprintf(fptr,
          "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
          "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n");
  const SE3 T_c_i = calib.T_i_c[0].inverse().cast<Scalar>();
  for (size_t cam_idx = 0; cam_idx < cameras_.size(); ++cam_idx) {
    if (image_id_to_cam_id[cam_idx + 1] != 1) {
      // We save only one pose per keyframe, taking the first camera as
      // reference
      continue;
    }

    const auto& cam = cameras_[cam_idx];
    const SE3 T_w_c = cam.T_c_w.inverse();
    const SE3 T_w_i = T_w_c * T_c_i;
    const Vec3 t = T_w_i.translation();
    const SO3 R = T_w_i.so3();
    const Eigen::Quaternion<Scalar> q = R.unit_quaternion();
    const size_t frame_id =
        sequential_colmap_id_to_basalt_id.count(cam_idx + 1)
            ? sequential_colmap_id_to_basalt_id.at(cam_idx + 1)
            : cam_idx;

    fprintf(fptr, "%lu,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", frame_id, t.x(), t.y(),
            t.z(), q.w(), q.x(), q.y(), q.z());
  }
  std::fclose(fptr);
  return true;
}

template <typename Scalar>
void BalProblem<Scalar>::normalize(const double new_scale) {
  // TODO: try out normalization mentioned in MCBA paper to see if it has
  // additional benefit on numerics (note that we already have jacobian scaling)

  // compute median point coordinates (x,y,z)
  std::vector<Scalar> tmp(num_landmarks());
  Vec3 median;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < num_landmarks(); ++i) {
      tmp[i] = landmarks_[i].p_w(j);
    }
    median(j) = median_destructive(tmp);
  }

  // compute median absolute deviation (l1-norm)
  for (int i = 0; i < num_landmarks(); ++i) {
    tmp[i] = (landmarks_[i].p_w - median).template lpNorm<1>();
  }
  const Scalar median_abs_deviation = median_destructive(tmp);

  // normalize scale to constant
  const Scalar scale = new_scale / median_abs_deviation;

  if (!quiet_) {
    LOG(INFO) << "Normalizing BAL problem (median: " << median.transpose()
              << ", MAD: " << median_abs_deviation << ", scale: " << scale
              << ")";
  }

  // update landmarks: X = scale * (X - median)
  for (auto& lm : landmarks_) {
    lm.p_w = scale * (lm.p_w - median);
  }

  // update cameras: center = scale * (center - median)
  for (auto& cam : cameras_) {
    SE3 T_w_c = cam.T_c_w.inverse();
    T_w_c.translation() = scale * (T_w_c.translation() - median);
    cam.T_c_w = T_w_c.inverse();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::filter_obs(const double threshold) {
  CHECK_GE(threshold, 0.0);

  if (threshold > 0) {
    if (!quiet_) {
      LOG(INFO) << "Filtering observations with z < {}"_format(threshold);
    }
  } else {
    return;
  }

  // Remove observations with depth of 3D point in camera frame closer than
  // threshold.
  for (auto& lm : landmarks_) {
    for (auto it = lm.obs.cbegin(); it != lm.obs.cend();) {
      const auto& cam = cameras_.at(it->first);
      Vec3 p3d_cam = cam.T_c_w * lm.p_w;

      if (p3d_cam.z() < threshold) {
        it = lm.obs.erase(it);
      } else {
        ++it;
      }
    }
  }

  Landmarks filtered_landmarks;

  // Filter landmarks with number of observations less than 2
  std::copy_if(landmarks_.begin(), landmarks_.end(),
               std::back_inserter(filtered_landmarks),
               [](const auto& lm) { return lm.obs.size() >= 2; });

  landmarks_ = std::move(filtered_landmarks);
}

template <typename Scalar>
void BalProblem<Scalar>::perturb(double rotation_sigma,
                                 double translation_sigma,
                                 double landmark_sigma, int seed) {
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);
  CHECK_GE(landmark_sigma, 0.0);

  if (rotation_sigma > 0 || translation_sigma > 0 || landmark_sigma > 0) {
    if (!quiet_) {
      LOG(INFO) << "Perturbing state (seed: {}): R: {}, t: {}, p: {}"
                   ""_format(seed, rotation_sigma, translation_sigma,
                             landmark_sigma);
    }
  }

  std::random_device r;
  std::default_random_engine eng =
      seed < 0
          ? std::default_random_engine{std::random_device{}()}
          : std::default_random_engine{
                static_cast<std::default_random_engine::result_type>(seed)};

  if (rotation_sigma > 0 || translation_sigma > 0) {
    for (auto& cam : cameras_) {
      // perturb camera center in world coordinates
      if (translation_sigma > 0) {
        SE3 T_w_c = cam.T_c_w.inverse();
        T_w_c.translation() += perturbation<Scalar, 3>(translation_sigma, eng);
        cam.T_c_w = T_w_c.inverse();
      }
      // local rotation perturbation in camera frame
      if (rotation_sigma > 0) {
        cam.T_c_w.so3() =
            SO3::exp(perturbation<Scalar, 3>(rotation_sigma, eng)) *
            cam.T_c_w.so3();
      }
    }
  }

  // perturb landmarks
  if (landmark_sigma > 0) {
    for (auto& lm : landmarks_) {
      lm.p_w += perturbation<Scalar, 3>(landmark_sigma, eng);
    }
  }
}

template <class Scalar>
void BalProblem<Scalar>::postprocress(const BalDatasetOptions& options,
                                      PipelineTimingSummary* timing_summary) {
  Timer t;

  if (options.save_output) {
    save_rootba(options.output_optimized_path);
  }

  if (options.save_bal != "") {
    save_bal(options.save_bal);
  }

  if (timing_summary) {
    timing_summary->postprocess_time = t.elapsed();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::copy_to_camera_state(VecX& camera_state) const {
  // Compute total size dynamically (pose 7 + intrinsics per cam).
  int total_size = 0;
  for (const auto& cam : cameras_) {
    total_size += 7 + cam.intrinsics.getN();
  }
  camera_state.resize(total_size);

  int offset = 0;
  for (const auto& cam : cameras_) {
    const int block_size = 7 + cam.intrinsics.getN();
    camera_state.segment(offset, block_size) = cam.params();
    offset += block_size;
  }
}

template <typename Scalar>
void BalProblem<Scalar>::copy_from_camera_state(const VecX& camera_state) {
  int offset = 0;
  for (auto& cam : cameras_) {
    const int block_size = 7 + cam.intrinsics.getN();
    CHECK_GE(camera_state.size(), offset + block_size);
    cam.from_params(camera_state.segment(offset, block_size));
    offset += block_size;
  }
  CHECK_EQ(offset, camera_state.size())
      << "Camera state vector size does not match accumulated block sizes.";
}

template <typename Scalar>
void BalProblem<Scalar>::backup() {
  for (auto& cam : cameras_) {
    cam.backup();
  }
  for (auto& lm : landmarks_) {
    lm.backup();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::restore() {
  for (auto& cam : cameras_) {
    cam.restore();
  }
  for (auto& lm : landmarks_) {
    lm.restore();
  }
}

template <typename Scalar>
int BalProblem<Scalar>::num_observations() const {
  int num = 0;
  for (auto& lm : landmarks_) {
    num += lm.obs.size();
  }
  return num;
}

template <typename Scalar>
int BalProblem<Scalar>::max_num_observations_per_lm() const {
  int num = 0;
  for (auto& lm : landmarks_) {
    num = std::max(num, static_cast<int>(lm.obs.size()));
  }
  return num;
}

namespace {  // helper

template <class T>
struct is_map {
  static constexpr bool value = false;
};

template <class Key, class Value>
struct is_map<std::map<Key, Value>> {
  static constexpr bool value = true;
};

// NOLINTNEXTLINE
struct default_initialized_atomic_bool : public std::atomic<bool> {
  default_initialized_atomic_bool() { store(false, std::memory_order_relaxed); }
};

}  // namespace

template <typename Scalar>
double BalProblem<Scalar>::compute_rcs_sparsity() const {
  const int num_cams = num_cameras();
  const int num_rcs_blocks = num_cams * num_cams;

  // Note: absl::flat_hash_set<int> is noticably faster than
  // std::unordered_set<int> and a lot faster than
  // std::unordered_set<pair<size_t, size_t>>. An array of bool is a lot
  // faster still, but might need a lot of memory for problems with many
  // cameras.

#if 0
  // absl::flat_hash_set<int> cam_pairs;
  Eigen::VectorX<bool> mask =
      Eigen::VectorX<bool>::Constant(num_rcs_blocks, false);

  for (const auto& lm : landmarks_) {
    for (const auto& [cam_idx_i, _] : lm.obs) {
      for (const auto& [cam_idx_j, _] : lm.obs) {
        if (cam_idx_j < cam_idx_i) {
          int index = cam_idx_i * num_cams + cam_idx_j;
          // cam_pairs.emplace(index);
          mask(index) = true;
        } else {
          // NOTE: the early abort with 'break' assumes ordered lm.obs
          static_assert(is_map<decltype(lm.obs)>::value);
          break;
        }
      }
    }
  }

  // const int num_non_zero_rcs_blocks = num_cams + 2 * cam_pairs.size();
  const int num_non_zero_rcs_blocks = num_cams + 2 * mask.count();
#else

  std::vector<default_initialized_atomic_bool> mask2(num_rcs_blocks);

  // TODO: verify that we really don't need memory barrier before and after
  // parallel for

  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      const auto& lm = landmarks_[r];
      for (const auto& [cam_idx_i, _] : lm.obs) {
        for (const auto& [cam_idx_j, _] : lm.obs) {
          if (cam_idx_j < cam_idx_i) {
            int index = cam_idx_i * num_cams + cam_idx_j;
            mask2[index].store(true, std::memory_order_relaxed);
          } else {
            // NOTE: the early abort with 'break' assumes ordered lm.obs
            static_assert(is_map<decltype(lm.obs)>::value);
            break;
          }
        }
      }
    }
  };

  tbb::blocked_range<size_t> range(0, landmarks_.size());
  tbb::parallel_for(range, body);

  const int num_non_zero_rcs_blocks =
      num_cams + 2 * std::count(mask2.begin(), mask2.end(), true);
#endif

  return 1. - num_non_zero_rcs_blocks / double(num_rcs_blocks);
}

template <class Scalar>
void BalProblem<Scalar>::summarize_problem(DatasetSummary& summary,
                                           bool compute_sparsity) const {
  summary.type = "bal";
  summary.num_cameras = num_cameras();
  summary.num_landmarks = num_landmarks();
  summary.num_observations = num_observations();

  if (compute_sparsity) {
    // can be a bit expensive for dense problems, so compute only when needed
    Timer timer;
    summary.rcs_sparsity = compute_rcs_sparsity();

    if (!quiet_) {
      // output runtime for this computation, b/c it can be quite large for
      // denser problems (so we notice when we should work in improving runtime)
      LOG(INFO) << "Computed RCS sparsity: {:.2f} ({:.3f}s)"_format(
          summary.rcs_sparsity, timer.elapsed());
    }
  }

  auto stats = [](const ArrXd& data) {
    DatasetSummary::Stats res;
    res.mean = data.mean();
    res.min = data.minCoeff();
    res.max = data.maxCoeff();
    res.stddev = std::sqrt((data - res.mean).square().sum() / data.size());
    return res;
  };

  // per landmark observation stats
  {
    ArrXd per_lm_obs(num_landmarks());
    for (int i = 0; i < num_landmarks(); ++i) {
      per_lm_obs(i) = landmarks_.at(i).obs.size();
    }
    summary.per_lm_obs = stats(per_lm_obs);
    CHECK_NEAR(summary.per_lm_obs.mean,
               double(num_observations()) / num_landmarks(), 1e-9);
  }

  // no per hostframe landmark stats
  summary.per_host_lms = DatasetSummary::Stats();
}

template <typename Scalar>
std::string BalProblem<Scalar>::stats_to_string() const {
  DatasetSummary summary;
  summarize_problem(summary, false);

  return "BAL problem stats: {} cams, {} lms, {} obs, per-lm-obs: "
         "{:.1f}+-{:.1f}/{}/{}"
         ""_format(num_cameras(), num_landmarks(), num_observations(),
                   summary.per_lm_obs.mean, summary.per_lm_obs.stddev,
                   int(summary.per_lm_obs.min), int(summary.per_lm_obs.max));
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary) {
  // random seed
  if (options.random_seed >= 0) {
    std::srand(options.random_seed);
  }

  Timer timer;

  // auto detect input type
  BalDatasetOptions::DatasetType input_type = options.input_type;
  if (BalDatasetOptions::DatasetType::AUTO == input_type) {
    input_type = autodetect_input_type(options.input);
    if (!options.quiet) {
      LOG(INFO) << "Autodetected input dataset type as {}."_format(
          wise_enum::to_string(input_type));
    }
  }

  // load dataset as double
  BalProblem<double> bal_problem;
  bal_problem.set_quiet(options.quiet);
  switch (input_type) {
    case BalDatasetOptions::DatasetType::ROOTBA:
      bal_problem.load_rootba(options.input);
      break;
    case BalDatasetOptions::DatasetType::BAL:
      bal_problem.load_bal(options.input);
      break;
    case BalDatasetOptions::DatasetType::BUNDLER:
      bal_problem.load_bundler(options.input);
      break;
    case BalDatasetOptions::DatasetType::COLMAP:
      bal_problem.load_colmap(options.input);
      break;
    default:
      LOG(FATAL) << "unreachable";
  }

  const double time_load = timer.reset();

  // normalize to fixed scale and center (as double, since there are some
  // overflow issues with float for large problems)
  if (options.normalize) {
    bal_problem.normalize(options.normalization_scale);
  }

  // perturb state if sigmas are positive
  bal_problem.perturb(options.rotation_sigma, options.translation_sigma,
                      options.point_sigma, options.random_seed);

  // Filter observations of points closer than threshold to the camera
  bal_problem.filter_obs(options.init_depth_threshold);

  // convert to Scalar if needed
  BalProblem<Scalar> res;
  if constexpr (std::is_same_v<Scalar, double>) {
    res = std::move(bal_problem);
  } else {
    res = bal_problem.copy_cast<Scalar>();
  }

  const double time_preprocess = timer.reset();

  if (timing_summary) {
    timing_summary->load_time = time_load;
    timing_summary->preprocess_time = time_preprocess;
  }

  if (dataset_summary) {
    dataset_summary->input_path = options.input;
    res.summarize_problem(*dataset_summary, true);
  }

  // print some info
  if (!options.quiet) {
    LOG(INFO) << res.stats_to_string();
  }

  return res;
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary) {
  BalDatasetOptions options;
  options.input = path;
  return load_normalized_bal_problem<Scalar>(options, dataset_summary,
                                             timing_summary);
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem_quiet(const std::string& path) {
  BalDatasetOptions options;
  options.input = path;
  options.quiet = true;
  return load_normalized_bal_problem<Scalar>(options);
}

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class BalProblem<float>;

template BalProblem<float> load_normalized_bal_problem<float>(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<float> load_normalized_bal_problem<float>(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<float> load_normalized_bal_problem_quiet<float>(
    const std::string& path);
#endif

// BalProblem in double is used by the ceres solver and GUI, so always
// compile it; it should not be a big compilation overhead.
// #ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class BalProblem<double>;

template BalProblem<double> load_normalized_bal_problem<double>(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<double> load_normalized_bal_problem<double>(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<double> load_normalized_bal_problem_quiet<double>(
    const std::string& path);
// #endif

}  // namespace rootba
