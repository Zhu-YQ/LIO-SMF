#pragma once

#include <deque>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;
using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;

struct MotionState {
    double time_stamp_;

    Vector3d p_;
    Vector3d v_;
    SO3d R_;

    Vector3d a_;
    Vector3d w_;

    MotionState(double time_stamp, const Vector3d& p, const Vector3d& v, const SO3d R, const Vector3d& a, const Vector3d& w)
        : time_stamp_(time_stamp)
        , p_(p)
        , v_(v)
        , R_(R)
        , a_(a)
        , w_(w)
    {
    }
};

struct MeasureGroup // Lidar data and imu dates for the curent process
{
    double lidar_begin_time;
    double lidar_end_time;

    PointCloudType::Ptr lidar;
    std::deque<sensor_msgs::Imu::Ptr> imu_buffer;

    MeasureGroup()
    {
        lidar_begin_time = 0.0;
        this->lidar.reset(new PointCloudType());
    };

    void preprocessPointCloud(const std::vector<MotionState>& motion_state_vec, const SO3d& R_imu_lidar, const Vector3d& t_imu_lidar)
    {
        if (lidar->points.begin() == lidar->points.end()) {
            return;
        }

        const SE3d T_lidar_imu = SE3d(R_imu_lidar, t_imu_lidar).inverse();
        const SO3d R_lidar_imu = T_lidar_imu.so3();
        const Vector3d t_lidar_imu = T_lidar_imu.translation();

        std::sort(lidar->points.begin(), lidar->points.end(),
            [](const PointType& x, const PointType& y) { return (x.curvature < y.curvature); });

        const auto t_w_imu_f = motion_state_vec.back().p_;
        const auto R_w_imu_f = motion_state_vec.back().R_;

        auto it_pcl = lidar->points.end() - 1;
        for (auto it_kp = motion_state_vec.end() - 1; it_kp != motion_state_vec.begin(); it_kp--) {
            auto head = it_kp - 1;
            auto tail = it_kp;

            const SO3d& R_imu = head->R_;
            const Vector3d& vel_imu = head->v_;
            const Vector3d& pos_imu = head->p_;

            const Vector3d& acc_imu = tail->a_;
            const Vector3d& angvel_avr = tail->w_;

            for (; head->time_stamp_ < (lidar_begin_time + 1.0e-3 * it_pcl->curvature)
                 && (lidar_begin_time + 1.0e-3 * it_pcl->curvature) < tail->time_stamp_;
                 it_pcl--) {

                /* do not undistort */
                // const Vector3d point_lidar(it_pcl->x, it_pcl->y, it_pcl->z);
                // const Vector3d point_imu_j = R_imu_lidar * point_lidar + t_imu_lidar;
                // it_pcl->x = point_imu_j[0];
                // it_pcl->y = point_imu_j[1];
                // it_pcl->z = point_imu_j[2];

                /* interpolation */
                // double query_time_stamp = lidar_begin_time + 1.0e-3 * it_pcl->curvature;
                // double dt = tail->time_stamp_ - head->time_stamp_;
                // double dt_prime = query_time_stamp - head->time_stamp_;
                // double s = dt_prime / dt;

                // // p
                // const auto p_head = head->p_;
                // const auto p_tail = tail->p_;
                // const Vector3d t_w_imu_j = p_head + s * (p_tail - p_head);

                // // q
                // const auto q_head = head->R_.unit_quaternion();
                // const auto q_tail = tail->R_.unit_quaternion();
                // const auto q_w_imu_j = q_head.slerp(s, q_tail);
                // const auto R_w_imu_j = q_w_imu_j.toRotationMatrix();

                // const Vector3d point_lidar(it_pcl->x, it_pcl->y, it_pcl->z);

                // const Vector3d point_imu_j = R_imu_lidar * point_lidar + t_imu_lidar;
                // const Vector3d point_w = R_w_imu_j * point_imu_j + t_w_imu_j;
                // const Vector3d point_imu_f = R_w_imu_f.inverse() * point_w - R_w_imu_f.inverse() * t_w_imu_j;

                // it_pcl->x = point_imu_f[0];
                // it_pcl->y = point_imu_f[1];
                // it_pcl->z = point_imu_f[2];

                /* backward propagation */
                double dt = it_pcl->curvature / double(1000) - head->time_stamp_;

                const Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);

                const SO3d R_i = R_imu * SO3d::exp(angvel_avr * dt);
                const Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - t_w_imu_f;

                const Vector3d P_compensate = R_w_imu_f.inverse() * (R_i * (R_imu_lidar * P_i + t_imu_lidar) + T_ei);

                it_pcl->x = P_compensate[0];
                it_pcl->y = P_compensate[1];
                it_pcl->z = P_compensate[2];

                if (it_pcl == lidar->begin()) {
                    break;
                }
            }
        }
    }
};

SO3d FromMat2SO3(const Matrix3d& R_mat)
{
    Eigen::Quaterniond R_q(R_mat);
    R_q.normalize();

    return SO3d(R_q);
}