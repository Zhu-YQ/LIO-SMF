#pragma once

#include <cmath>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <sophus/so3.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include "ellipse.h"

#include "../bounding/bounding.h"
#include "../map/map_manager.h"

using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using MatrixXd = Eigen::MatrixXd;

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Quatd = Eigen::Quaterniond;

using ES1 = ElpSet<1>;
using ES3 = ElpSet<3>;
using ES4 = ElpSet<4>;
using ES6 = ElpSet<6>;

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

struct ResultTuple {
    Eigen::Matrix<double, 1, 6> J;
    Matrix6d JTJ;
    Vector6d JTr;

    ResultTuple()
    {
        J.setZero();
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple& other)
    {
        this->J += other.J;
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }
};

struct ESSMF {
    /* state */
    // nominal
    Vector3d t_;
    Vector3d v_;
    SO3d R_;
    Vector3d ba_;
    Vector3d bg_;
    Vector3d g_;

    // error
    ES3 d_t_; // translational local protection level
    ES3 d_v_;
    ES3 d_theta_; // rotational local protection level

    ES3 d_ba_;
    ES3 d_bg_;

    ES3 d_t_global_; // translational global protection level
    ES3 d_theta_global_; // rotational global protection level

    /* last imu measurements */
    Vector3d a_last_;
    Vector3d w_last_;

    /* update */
    std::vector<Vector3d> icp_source_vec_;
    std::vector<Vector4d> icp_plane_ud_vec_;
    std::vector<PointVector> icp_plane_pts_vec_;
    std::vector<ES3> icp_point_error_es_;

    std::vector<double> icp_distances_vec_;
    std::vector<Eigen::Matrix<double, 1, 6>> icp_J_vec_;

    bool valid_update_;

    int max_iteration_;
    double converge_thresh_;

    ES3 t_es_observed_last_;
    double t_es_observed_last_time_stamp_;

    double range_bound_;
    double bearing_bound_;

    SO3d R_icp_;
    Vector3d t_icp_;

    // degeneracy
    bool deg_process_en_;

    bool degeneracy_t_;
    bool degeneracy_r_;

    double k_abs_t_;
    double k_rel_t_;

    double k_abs_r_;
    double k_rel_r_;

    /* compensation */
    double point_compensation_;

    double bch_compensation_;

    double map_anchor_compensation_t_;
    double map_anchor_compensation_r_;

    double global_uncertainty_compensation_t_;
    double global_uncertainty_compensation_r_;

    double lumped_trans_;
    double lumped_rot_;

    /* system */
    double time_stamp_;

    SO3d R_imu_lidar_;
    Vector3d t_imu_lidar_;

    /* map */
    ES3 d_t_map_;
    ES3 d_theta_map_;

    SE3d T_map_anchor_;

    SO3d R_last_key_frame_;
    Vector3d t_last_key_frame_;

    MapManager map_;
    PointCloudType visual_map_; // map for visualization
    double local_map_distance_;

    /* exp record */
    double icp_duration_;
    double uncertainty_duration_;
    double intersection_duration_;

    ESSMF()
        : t_(Vector3d::Zero())
        , v_(Vector3d::Zero())
        , R_()
        , ba_(Vector3d::Zero())
        , bg_(Vector3d::Zero())
        , g_(Vector3d::Zero())
        , d_t_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , d_v_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , d_theta_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , d_ba_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , d_bg_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , d_t_global_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , d_theta_global_(Vector3d::Zero(), 0.1 * Matrix3d::Identity())
        , time_stamp_(-1)
        , max_iteration_(30)
        , converge_thresh_(10000000)
        , t_es_observed_last_()
        , t_es_observed_last_time_stamp_(-1)
        , range_bound_(0.05)
        , bearing_bound_(0.2)
        , d_t_map_(Vector3d::Zero(), 0.001 * Matrix3d::Identity())
        , d_theta_map_(Vector3d::Zero(), 0.001 * Matrix3d::Identity())
        , local_map_distance_(150)
        , point_compensation_(0.0001)
        , bch_compensation_(0.01)
        , map_anchor_compensation_t_(0.001)
        , map_anchor_compensation_r_(0.001)
        , global_uncertainty_compensation_t_(0.001)
        , global_uncertainty_compensation_r_(0.001)
        , lumped_trans_(0.2)
        , lumped_rot_(0.5)
        , a_last_(Vector3d::Zero())
        , w_last_(Vector3d::Zero())
        , valid_update_(true)
        , deg_process_en_(false)
        , degeneracy_t_(false)
        , degeneracy_r_(false)
        , k_abs_t_(0)
        , k_rel_t_(0)
        , k_abs_r_(0)
        , k_rel_r_(0)
        , icp_duration_(0)
        , uncertainty_duration_(0)
        , intersection_duration_(0) {};

    void setBias(const Vector3d& ba, const Matrix3d& N_ba, const Vector3d& bg, const Matrix3d& N_bg);
    void setGravity(const Vector3d& g);
    void setTime(double time_stamp);

    void alignGravity();

    Vector3d getPosition();
    Quatd getOrientation();
    nav_msgs::Odometry getPose();
    sensor_msgs::PointCloud2::ConstPtr getMapPointCloud();

    void predict(const sensor_msgs::Imu::ConstPtr& imu_msg, const Matrix3d& N_a, const Matrix3d& N_g);
    ES3 calPointNoiseES(const Vector3d& p_imu);
    bool zeroVelocityUpdate(double velocity_bound);
    void updateNominalState();
    void updateErrorState(const PointCloudType::ConstPtr& current_point_cloud, double lidar_time_stamp);

    void updateMap(const PointCloudType::ConstPtr& current_point_cloud);
};
