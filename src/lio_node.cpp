#include <deque>
#include <iostream>
#include <filesystem>

#include <Eigen/Core>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <omp.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

#include "filter/es_smf.h"
#include "filter/preprocess.h"
#include "filter/static_imu_init.h"
#include "utils/log_helper.h"
#include "utils/utility.h"

/* Sensor Msg */
std::deque<double> time_buffer;
MeasureGroup new_meas_group;

// IMU
std::deque<sensor_msgs::Imu::Ptr> imu_buffer;
sensor_msgs::Imu::Ptr last_imu_msg;
double time_diff_lidar_to_imu = 0;
double last_timestamp_imu = -1;
bool gravity_align = false;
void ImuMsgCallback(const sensor_msgs::Imu::Ptr& msg)
{
    sensor_msgs::Imu::Ptr msg_out(new sensor_msgs::Imu(*msg));
    msg_out->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - time_diff_lidar_to_imu);

    // std::cout << msg->header.stamp.toSec() << "\n";

    if (msg_out->header.stamp.toSec() < last_timestamp_imu) {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
        last_imu_msg.reset();
    }

    imu_buffer.push_back(msg_out);
    last_timestamp_imu = msg_out->header.stamp.toSec();
}

// LiDAR
std::shared_ptr<Preprocess> p_pre(new Preprocess());
std::deque<PointCloudType::Ptr> lidar_buffer;
void StdLiDARCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    PointCloudType::Ptr pcl_ptr(new PointCloudType());
    p_pre->process(msg, pcl_ptr);

    lidar_buffer.push_back(pcl_ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
}

void LivoxLiDARCallback(const livox_ros_driver::CustomMsg::ConstPtr& msg)
{
    PointCloudType::Ptr pcl_ptr(new PointCloudType());
    p_pre->process(msg, pcl_ptr);

    lidar_buffer.push_back(pcl_ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
}

bool lidar_pushed = false;
double lidar_mean_scantime = 0;
double lidar_end_time = 0;
int scan_num = 0;

sensor_msgs::Imu::Ptr InterpolateIMUMsg(const sensor_msgs::Imu::Ptr& imu_before, const sensor_msgs::Imu::Ptr& imu_after, double target_time)
{
    if (!imu_before || !imu_after) {
        return nullptr;
    }

    const double time_before = imu_before->header.stamp.toSec();
    const double time_after = imu_after->header.stamp.toSec();
    if (time_after <= time_before || target_time < time_before || target_time > time_after) {
        return nullptr;
    }

    const double ratio = (target_time - time_before) / (time_after - time_before);
    sensor_msgs::Imu::Ptr interpolated(new sensor_msgs::Imu(*imu_before));
    interpolated->header.stamp = ros::Time().fromSec(target_time);

    interpolated->linear_acceleration.x = imu_before->linear_acceleration.x + ratio * (imu_after->linear_acceleration.x - imu_before->linear_acceleration.x);
    interpolated->linear_acceleration.y = imu_before->linear_acceleration.y + ratio * (imu_after->linear_acceleration.y - imu_before->linear_acceleration.y);
    interpolated->linear_acceleration.z = imu_before->linear_acceleration.z + ratio * (imu_after->linear_acceleration.z - imu_before->linear_acceleration.z);

    interpolated->angular_velocity.x = imu_before->angular_velocity.x + ratio * (imu_after->angular_velocity.x - imu_before->angular_velocity.x);
    interpolated->angular_velocity.y = imu_before->angular_velocity.y + ratio * (imu_after->angular_velocity.y - imu_before->angular_velocity.y);
    interpolated->angular_velocity.z = imu_before->angular_velocity.z + ratio * (imu_after->angular_velocity.z - imu_before->angular_velocity.z);

    return interpolated;
}

bool SyncMsgs()
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /* push a lidar scan */
    if (!lidar_pushed) {
        new_meas_group.lidar = lidar_buffer.front();
        new_meas_group.lidar_begin_time = time_buffer.front();

        if (new_meas_group.lidar->points.size() <= 1) {
            lidar_end_time = new_meas_group.lidar_begin_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        } else if (new_meas_group.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) {
            lidar_end_time = new_meas_group.lidar_begin_time + lidar_mean_scantime;
        } else {
            scan_num++;
            lidar_end_time = new_meas_group.lidar_begin_time + new_meas_group.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (new_meas_group.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        new_meas_group.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) {
        return false;
    }

    new_meas_group.imu_buffer.clear();
    while (!imu_buffer.empty() && imu_buffer.front()->header.stamp.toSec() <= lidar_end_time) {
        last_imu_msg = imu_buffer.front();
        new_meas_group.imu_buffer.push_back(last_imu_msg);
        imu_buffer.pop_front();
    }

    if (last_imu_msg && last_imu_msg->header.stamp.toSec() < lidar_end_time) {
        if (imu_buffer.empty()) {
            return false;
        }

        sensor_msgs::Imu::Ptr imu_at_lidar_end
            = InterpolateIMUMsg(last_imu_msg, imu_buffer.front(), lidar_end_time);
        if (!imu_at_lidar_end) {
            ROS_WARN("Failed to interpolate IMU measurement at LiDAR scan end.");
            return false;
        }

        new_meas_group.imu_buffer.push_back(imu_at_lidar_end);
        last_imu_msg = imu_at_lidar_end;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;

    if (new_meas_group.imu_buffer.empty()) {
        return false;
    }

    return true;
}

// ellipsoidal set-membership filter
ESSMF filter;

double init_static_span = 5;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_node");
    ros::NodeHandle nh;

    /* I/O configs */
    tf::TransformBroadcaster br;
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/lio_smf/odom", 100);
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("/est_path", 20);

    // for visualization
    size_t odom_slow_counter = 0;
    ros::Publisher odom_pub_slow = nh.advertise<nav_msgs::Odometry>("/lio_smf/odom_slow", 100);

    ros::Publisher current_point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lio_smf/current_point_cloud", 10);
    ros::Publisher map_point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lio_smf/map_point_cloud", 10);

    // IMU
    std::string imu_topic;
    nh.param<std::string>("imu/topic", imu_topic, "");
    ros::Subscriber imu_sub = nh.subscribe(imu_topic, 200000, ImuMsgCallback);

    double acc_bound, ba_bound, gyr_bound, bg_bound = 0;
    nh.param<double>("imu/acc_bound", acc_bound, 0);
    nh.param<double>("imu/ba_bound", ba_bound, 0);
    nh.param<double>("imu/gyr_bound", gyr_bound, 0);
    nh.param<double>("imu/bg_bound", bg_bound, 0);

    nh.param<double>("imu/init_static_span", init_static_span, 0.1);
    nh.param<bool>("imu/gravity_align", gravity_align, false);

    // Stationary detection and zero-velocity update
    bool stationary_detection_enabled;
    double stationary_gyro_mean_threshold;
    double stationary_gyro_std_threshold;
    double stationary_acc_norm_threshold;
    double stationary_acc_std_threshold;
    double stationary_lidar_translation_threshold;
    double stationary_lidar_rotation_threshold_deg;
    double stationary_velocity_bound;
    int stationary_enter_frames;
    int stationary_exit_frames;

    nh.param<bool>("stationary/enabled", stationary_detection_enabled, true);
    nh.param<double>("stationary/gyro_mean_threshold", stationary_gyro_mean_threshold, 0.03);
    nh.param<double>("stationary/gyro_std_threshold", stationary_gyro_std_threshold, 0.01);
    nh.param<double>("stationary/acc_norm_threshold", stationary_acc_norm_threshold, 0.25);
    nh.param<double>("stationary/acc_std_threshold", stationary_acc_std_threshold, 0.15);
    nh.param<double>("stationary/lidar_translation_threshold", stationary_lidar_translation_threshold, 0.02);
    nh.param<double>("stationary/lidar_rotation_threshold_deg", stationary_lidar_rotation_threshold_deg, 0.5);
    nh.param<double>("stationary/velocity_bound", stationary_velocity_bound, 0.03);
    nh.param<int>("stationary/enter_frames", stationary_enter_frames, 3);
    nh.param<int>("stationary/exit_frames", stationary_exit_frames, 2);

    // LiDAR Config
    std::string lidar_topic;
    nh.param<std::string>("lidar/topic", lidar_topic, "");
    nh.param<double>("lidar/blind_min", p_pre->blind_min, 0.01);
    nh.param<double>("lidar/blind_max", p_pre->blind_max, 10000);
    nh.param<int>("lidar/type", p_pre->lidar_type, LIVOX);
    nh.param<int>("lidar/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("lidar/timestamp_unit", p_pre->time_unit, US);

    nh.param<double>("lidar/range_bound", filter.range_bound_, 0.05);
    nh.param<double>("lidar/bearing_bound", filter.bearing_bound_, 0.2);

    // ICP
    nh.param<bool>("icp/feature_enabled", p_pre->feature_enabled, false);
    nh.param<int>("icp/point_filter_num", p_pre->point_filter_num, 4);
    nh.param<int>("icp/max_iteration", filter.max_iteration_, 30);
    nh.param<double>("icp/converge_thresh", filter.converge_thresh_, 10000000);

    nh.param<bool>("icp/deg_process_en", filter.deg_process_en_, false);
    nh.param<double>("icp/k_abs_t", filter.k_abs_t_, 0);
    nh.param<double>("icp/k_rel_t", filter.k_rel_t_, 0);
    nh.param<double>("icp/k_abs_r", filter.k_abs_r_, 0);
    nh.param<double>("icp/k_rel_r", filter.k_rel_r_, 0);

    nh.param<double>("local_map_distance", filter.local_map_distance_, 150);
    nh.param<double>("icp/lumped_trans", filter.lumped_trans_, 0.2);
    nh.param<double>("icp/lumped_rot", filter.lumped_rot_, 0.5);

    ros::Subscriber lidar_sub = p_pre->lidar_type == LIVOX ? nh.subscribe(lidar_topic, 200000, LivoxLiDARCallback) : nh.subscribe(lidar_topic, 2000, StdLiDARCallback);

    // T_imu_lidar
    std::vector<double> extrin_t(3, 0.0);
    std::vector<double> extrin_R(9, 0.0);
    nh.param<vector<double>>("extrinsic_t", extrin_t, vector<double>());
    nh.param<vector<double>>("extrinsic_R", extrin_R, vector<double>());

    nh.param<double>("time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0);

    Matrix3d R_imu_lidar_mat(Matrix3d::Identity());
    R_imu_lidar_mat << MAT_FROM_ARRAY(extrin_R);

    SO3d R_imu_lidar = FromMat2SO3(R_imu_lidar_mat);
    Vector3d t_imu_lidar(VEC_FROM_ARRAY(extrin_t));

    // save
    std::string save_path;
    nh.param<std::string>("save_path", save_path, "");

    std::filesystem::path dir = save_path;
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    LogHelper pose_log_tum(save_path + "/est_traj.txt");
    pose_log_tum.open();
    LogHelper pose_log_ntuviral(save_path + "/est_traj.csv");
    pose_log_ntuviral.open();

    LogHelper shape_mat_t_log(save_path + "/est_extrema_t.txt");
    shape_mat_t_log.open();
    LogHelper shape_mat_theta_log(save_path + "/est_extrema_theta.txt");
    shape_mat_theta_log.open();

#ifdef EXP_LOG
    LogHelper time_log(save_path + "/time_record.txt");
    time_log.open();

    LogHelper valid_pts_log(save_path + "/valid_pts.txt");
    valid_pts_log.open();
#endif

    /* Core */
    // variables for initialization of IMU
    StaticIMUInit imu_init(init_static_span);

    bool has_last_lidar_pose = false;
    bool stationary = false;
    int stationary_candidate_count = 0;
    int moving_candidate_count = 0;
    SO3d last_lidar_R;
    Vector3d last_lidar_t = Vector3d::Zero();

    Eigen::Vector3d g(0, 0, 0);
    Eigen::Vector3d ba(0, 0, 0);
    Eigen::Vector3d bg(0, 0, 0);

    Matrix3d N_a, N_g; // shape matrices of UBB noises
    Matrix3d N_ba, N_bg; // shape matrices of imu biases

    SO3d R_imu_s_u;

    nav_msgs::Path path;
    path.header.frame_id = "loc_init";

    while (ros::ok()) {
        ros::spinOnce();

        if (imu_buffer.empty()) {
            // ROS_INFO("No sensor data.");
            continue;
        }

        // IMU initialization
        if (!imu_init.done()) {
            imu_init.addIMUMsg(imu_buffer.front());
            last_imu_msg = imu_buffer.front();
            imu_buffer.pop_front();
            imu_init.init();

            if (imu_init.done()) {
                g = imu_init.getG();
                ba = imu_init.getBa();
                bg = imu_init.getBg();

                // std::cout << "---------" << '\n';
                // std::cout << "IMU params: " << '\n';
                // std::cout << "g: " << g.transpose() << '\n';
                // std::cout << "ba: " << ba.transpose() << '\n';
                // std::cout << "bg: " << bg.transpose() << '\n';
                // std::cout << "acc_bound: " << acc_bound << ", ba_bound: " << ba_bound << '\n';
                // std::cout << "gyr_bound: " << gyr_bound << ", bg_bound: " << bg_bound << '\n';
                // std::cout << "---------" << '\n';

                ROS_INFO("IMU initialized!");

                N_a = 3.0 * std::pow(acc_bound, 2) * Matrix3d::Identity();
                N_g = 3.0 * std::pow(gyr_bound, 2) * Matrix3d::Identity();

                N_ba = 3.0 * std::pow(ba_bound, 2) * Matrix3d::Identity();
                N_bg = 3.0 * std::pow(bg_bound, 2) * Matrix3d::Identity();

                filter.setBias(ba, N_ba, bg, N_bg);
                filter.setGravity(g);
                filter.time_stamp_ = imu_init.done_time_stamp_;

                filter.R_imu_lidar_ = R_imu_lidar;
                filter.t_imu_lidar_ = t_imu_lidar;

                if (gravity_align) {
                    filter.alignGravity();
                }
            }

            continue;
        }

        if (!SyncMsgs()) {
            continue;
        }

        // IMU
        std::vector<MotionState> motion_state_vec; // save for undistortion

        Vector3d avg_a = Vector3d::Zero();
        Vector3d avg_w = Vector3d::Zero();

        for (size_t i = 0; i < new_meas_group.imu_buffer.size(); i++) {
            const auto imu_msg = new_meas_group.imu_buffer[i];
            Vector3d a(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
            Vector3d w(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
            a -= filter.ba_;
            w -= filter.bg_;

            avg_a += a;
            avg_w += w;
        }

        avg_a *= (1.0 / new_meas_group.imu_buffer.size());
        avg_w *= (1.0 / new_meas_group.imu_buffer.size());

        double var_a = 0;
        double var_w = 0;

        for (size_t i = 0; i < new_meas_group.imu_buffer.size(); i++) {
            const auto imu_msg = new_meas_group.imu_buffer[i];
            Vector3d a(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
            Vector3d w(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
            a -= filter.ba_;
            w -= filter.bg_;

            var_a += (a - avg_a).squaredNorm();
            var_w += (w - avg_w).squaredNorm();
        }

        const double acc_std = std::sqrt(var_a / new_meas_group.imu_buffer.size());
        const double gyro_std = std::sqrt(var_w / new_meas_group.imu_buffer.size());
        const bool imu_stationary_candidate
            = avg_w.norm() < stationary_gyro_mean_threshold
            && gyro_std < stationary_gyro_std_threshold
            && std::fabs(avg_a.norm() - filter.g_.norm()) < stationary_acc_norm_threshold
            && acc_std < stationary_acc_std_threshold;

        double predict_duration = 0;
        double update_duration = 0;

        // Prediction, deskewing and LiDAR update must also run while stationary.
        for (size_t i = 0; i < new_meas_group.imu_buffer.size(); i++) {
            auto& imu_msg = new_meas_group.imu_buffer[i];

            Vector3d a(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
            a -= filter.ba_;

            Vector3d w(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
            w -= filter.bg_;

            const auto predict_start_time = std::chrono::high_resolution_clock::now();
            filter.predict(imu_msg, N_a, N_g);
            const auto predict_end_time = std::chrono::high_resolution_clock::now();
            predict_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(predict_end_time - predict_start_time).count() * 1e-9;

            motion_state_vec.emplace_back(imu_msg->header.stamp.toSec(), filter.t_, filter.v_, filter.R_, a, w);
        }

        // LiDAR
        new_meas_group.preprocessPointCloud(motion_state_vec, R_imu_lidar, t_imu_lidar);

        const auto update_start_time = std::chrono::high_resolution_clock::now();
        filter.updateErrorState(new_meas_group.lidar, new_meas_group.lidar_end_time);

        bool lidar_stationary_candidate = false;
        const bool lidar_pose_reliable = filter.valid_update_ && !filter.degeneracy_t_ && !filter.degeneracy_r_;
        if (lidar_pose_reliable && has_last_lidar_pose) {
            const double translation_increment = (filter.t_icp_ - last_lidar_t).norm();
            const double rotation_increment = (last_lidar_R.inverse() * filter.R_icp_).log().norm();
            const double rotation_threshold = stationary_lidar_rotation_threshold_deg * M_PI / 180.0;

            lidar_stationary_candidate = translation_increment < stationary_lidar_translation_threshold && rotation_increment < rotation_threshold;
        }

        if (lidar_pose_reliable) {
            last_lidar_t = filter.t_icp_;
            last_lidar_R = filter.R_icp_;
            has_last_lidar_pose = true;
        } else {
            has_last_lidar_pose = false;
        }

        const bool stationary_candidate = stationary_detection_enabled && imu_stationary_candidate && lidar_stationary_candidate;
        if (stationary_candidate) {
            stationary_candidate_count++;
            moving_candidate_count = 0;
            if (stationary_candidate_count >= std::max(1, stationary_enter_frames)) {
                stationary = true;
            }
        } else {
            stationary_candidate_count = 0;
            moving_candidate_count++;
            if (moving_candidate_count >= std::max(1, stationary_exit_frames)) {
                stationary = false;
            }
        }

        if (stationary && stationary_candidate && !filter.zeroVelocityUpdate(stationary_velocity_bound)) {
            ROS_WARN_THROTTLE(1.0, "Zero-velocity set update rejected.");
        }

        filter.updateNominalState();
        const auto update_end_time = std::chrono::high_resolution_clock::now();
        update_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(update_end_time - update_start_time).count() * 1e-9;

        filter.updateMap(new_meas_group.lidar);

        /* Publish */
        // current point cloud
        sensor_msgs::PointCloud2::Ptr current_point_cloud_msg(new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*new_meas_group.lidar, *current_point_cloud_msg);
        current_point_cloud_msg->header.frame_id = "body";
        current_point_cloud_msg->header.stamp = ros::Time(new_meas_group.lidar_end_time);
        current_point_cloud_pub.publish(*current_point_cloud_msg);

        // map
        map_point_cloud_pub.publish(*filter.getMapPointCloud());

        // odom
        nav_msgs::Odometry odom_msg = filter.getPose();
        odom_pub.publish(odom_msg);

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z));
        tf::Quaternion q(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odom_msg.header.stamp, "loc_init", "body"));

        odom_slow_counter++;
        if (odom_slow_counter % 30 == 0) {
            odom_pub_slow.publish(odom_msg);
        }

        // path
        geometry_msgs::PoseStamped pose_stamped;

        pose_stamped.pose.position.x = odom_msg.pose.pose.position.x;
        pose_stamped.pose.position.y = odom_msg.pose.pose.position.y;
        pose_stamped.pose.position.z = odom_msg.pose.pose.position.z;
        pose_stamped.pose.orientation.w = 1;

        pose_stamped.header.stamp = odom_msg.header.stamp;
        pose_stamped.header.frame_id = "loc_init";

        path.header.stamp = odom_msg.header.stamp;
        path.poses.push_back(pose_stamped);
        path_pub.publish(path);

        /* Save */
        pose_log_tum.writePoseTum(odom_msg);
        pose_log_ntuviral.writePoseNtuviral(odom_msg);

        // local
        // shape_mat_t_log.write(odom_msg.header.stamp.toSec(), filter.d_t_.P_.data(), 9);
        // shape_mat_theta_log.write(odom_msg.header.stamp.toSec(), filter.d_theta_.P_.data(), 9);

        // global
        shape_mat_t_log.write(odom_msg.header.stamp.toSec(), filter.d_t_global_.P_.data(), 9);
        shape_mat_theta_log.write(odom_msg.header.stamp.toSec(), filter.d_theta_global_.P_.data(), 9);

#ifdef EXP_LOG
        double time_vals[] = { predict_duration, update_duration, filter.icp_duration_, filter.uncertainty_duration_, filter.intersection_duration_ };
        time_log.write(odom_msg.header.stamp.toSec(), time_vals, 5);

        valid_pts_log.write(odom_msg.header.stamp.toSec(), filter.icp_source_vec_.size());
#endif
    }

    pose_log_tum.close();
    pose_log_ntuviral.close();

    shape_mat_t_log.close();
    shape_mat_theta_log.close();

#ifdef EXP_LOG
    time_log.close();
    valid_pts_log.close();
    L_vec_log.close();
#endif

    /* Save PCD */
    // filter.map_.full_map_.ikd_tree_->flatten(filter.map_.full_map_.ikd_tree_->Root_Node, filter.map_.full_map_.ikd_tree_->PCL_Storage, NOT_RECORD);
    // PointCloudType::Ptr featsFromMap(new PointCloudType());
    // featsFromMap->points = filter.map_.full_map_.ikd_tree_->PCL_Storage;

    // pcl::PCDWriter pcd_writer;
    // pcd_writer.writeBinary(save_path + "/map.pcd", *featsFromMap);
}
