#pragma once

#include <iostream>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Eigenvalues>

#include <sophus/se3.hpp>

#include <pcl/point_types.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using Matrix4x3d = Eigen::Matrix<double, 4, 3>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;

using SE3d = Sophus::SE3d;
using SO3d = Sophus::SO3d;

using Vector3dVector = std::vector<Eigen::Vector3d>;
using Vector4dVector = std::vector<Eigen::Vector4d>;

using PointType = pcl::PointXYZINormal;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix4x3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix6d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Matrix6x3d)

/* icp */
// H = d2_Ji / d2_d_zeta
Matrix6d CalICPHessianWrtPose(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec);

// Hi = d_f / d_pi
std::vector<Matrix6x3d> CalICPDerivWrtSrcPt(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec, const Matrix6d& d2_J_d2_d_zeta_inv);

// Hi = d_f / d_ui
std::vector<Matrix6x3d> CalICPDerivWrtNormVec(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec, const Matrix6d& d2_J_d2_d_zeta_inv);

// Hi = d_f / d_di
std::vector<Vector6d> CalICPDerivWrtD(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec, const Matrix6d& d2_J_d2_dzeta_inv);
