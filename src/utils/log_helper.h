#pragma once

#include <fstream>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <sstream>
#include <string>

class LogHelper {
private:
    std::string file_path;
    std::ofstream fs;
    int counter_for_ntuviral_;

public:
    LogHelper(const std::string& path)
        : file_path(path)
        , fs()
        , counter_for_ntuviral_(0) {};

    void open();
    void close();

    template <typename T>
    void write(double stamp, T data)
    {
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);
        ss.precision(9);
        ss << stamp << " ";
        ss << data << '\n';

        fs << ss.str();
    }

    // void write(double stamp, const std::string& data);
    void write(double stamp, double* data, size_t len);

    void writePoseTum(const nav_msgs::Odometry& odom_msg);
    void writePoseNtuviral(const nav_msgs::Odometry& odom_msg);
};
