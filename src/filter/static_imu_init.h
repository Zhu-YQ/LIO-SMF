/* modified from https://github.com/gaoxiang12/slam_in_autonomous_driving */

#pragma once

#include <Eigen/Core>
#include <deque>
#include <sensor_msgs/Imu.h>

struct StaticIMUInit {
    std::deque<sensor_msgs::Imu::ConstPtr> imu_msg_queue;

    double static_span_;
    bool init_done_;
    double done_time_stamp_;

    Eigen::Vector3d bg_;
    Eigen::Vector3d ba_;
    Eigen::Vector3d gravity_;

    StaticIMUInit(double static_span)
        : static_span_(static_span)
        , init_done_(false)
        , done_time_stamp_(-1)
        , imu_msg_queue()
        , bg_(Eigen::Vector3d::Zero())
        , ba_(Eigen::Vector3d::Zero())
        , gravity_(Eigen::Vector3d::Zero())
    {
    }

    Eigen::Vector3d getBg() const { return bg_; }

    Eigen::Vector3d getBa() const { return ba_; }

    Eigen::Vector3d getG() const { return gravity_; }

    void addIMUMsg(const sensor_msgs::Imu::ConstPtr& msg) { imu_msg_queue.push_back(msg); }

    bool done() const { return init_done_; }

    void init()
    {
        if (imu_msg_queue.size() == 2) {
            return;
        }

        const double start_time = imu_msg_queue.front()->header.stamp.toSec();
        const double latest_time = imu_msg_queue.back()->header.stamp.toSec();
        if ((latest_time - start_time) < static_span_) {
            return;
        }

        Eigen::Vector3d mean_a = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_w = Eigen::Vector3d::Zero();

        for (const auto& msg : imu_msg_queue) {
            const Eigen::Vector3d a(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            mean_a += a;
            const Eigen::Vector3d w(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
            mean_w += w;
        }

        mean_a /= imu_msg_queue.size();
        mean_w /= imu_msg_queue.size();

        gravity_ = -(9.81) * mean_a / mean_a.norm();

        ba_ = mean_a + gravity_;
        bg_ = mean_w;

        init_done_ = true;
        done_time_stamp_ = latest_time;
        imu_msg_queue.clear();
    }
};
