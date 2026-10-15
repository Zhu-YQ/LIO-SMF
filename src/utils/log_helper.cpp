#include "log_helper.h"

void LogHelper::open()
{
    fs.open(file_path, std::ios::out);
    fs.setf(std::ios::fixed, std::ios::floatfield);
    fs.precision(16);
}

void LogHelper::close()
{
    fs.close();
}

// void LogHelper::write(double stamp, const std::string& data)
// {
//     std::stringstream ss;
//     ss.setf(std::ios::fixed, std::ios::floatfield);
//     ss.precision(9);
//     ss << stamp << " ";
//     ss << data << '\n';

//     fs << ss.str();
// }

void LogHelper::write(double stamp, double* data, size_t len)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(9);
    ss << stamp << " ";

    ss.precision(16);
    for (size_t i = 0; i < len; i++) {
        ss << data[i];
        if (i != len - 1) {
            ss << " ";
        }
    }
    ss << '\n';

    fs << ss.str();
}

void LogHelper::writePoseTum(const nav_msgs::Odometry& odom_msg)
{
    // TUM format: time t_x t_y t_z q_x q_y q_z q_w
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(9);
    ss << odom_msg.header.stamp.toSec() << " ";

    ss.precision(16);
    ss << odom_msg.pose.pose.position.x << " " << odom_msg.pose.pose.position.y << " " << odom_msg.pose.pose.position.z << " ";
    ss << odom_msg.pose.pose.orientation.x << " " << odom_msg.pose.pose.orientation.y << " " << odom_msg.pose.pose.orientation.z << " " << odom_msg.pose.pose.orientation.w;
    ss << '\n';

    fs << ss.str();
}

void LogHelper::writePoseNtuviral(const nav_msgs::Odometry& odom_msg)
{
    // NTU VIRAL format:
    // time, field.header.seq, field.header.stamp,
    // pose.position.x, pose.position.y, pose.position.z,
    // pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
    // twist.linear.x, twist.linear.y, twist.linear.z,
    // twist.angular.x, twist.angular.y, twist.angular.z

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(9);
    ss << odom_msg.header.stamp.toSec() * 1.0e9 << ", ";
    ss << counter_for_ntuviral_ << ", ";
    ss << odom_msg.header.stamp.toSec() << ", ";

    ss.precision(16);
    ss << odom_msg.pose.pose.position.x << ", " << odom_msg.pose.pose.position.y << ", " << odom_msg.pose.pose.position.z << ", ";
    ss << odom_msg.pose.pose.orientation.x << ", " << odom_msg.pose.pose.orientation.y << ", " << odom_msg.pose.pose.orientation.z << ", " << odom_msg.pose.pose.orientation.w << ", ";
    ss << "0, 0, 0, 0, 0, 0\n";

    fs << ss.str();

    counter_for_ntuviral_++;
}
