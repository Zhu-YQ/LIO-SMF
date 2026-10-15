#pragma once

#include <deque>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

#include "./ikd_tree/ikd_Tree.h"

using Vector3d = Eigen::Vector3d;
using SE3d = Sophus::SE3d;

using PointType = pcl::PointXYZINormal;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

struct SubMap {
    std::unique_ptr<KD_TREE<PointType>> ikd_tree_;
    
    SE3d T_w_sub_;

    SubMap()
        : ikd_tree_(new KD_TREE<PointType>())
        , T_w_sub_()
    {}

    int size() const
    {
        return ikd_tree_->size();
    }
};

struct MapManager {
    SubMap full_map_;

    std::vector<SubMap> sub_map_vec_;
    size_t current_sub_map_id_;

    MapManager()
        : current_sub_map_id_(-1) {};

    bool empty() const
    {
        return current_sub_map_id_ == -1;
    }

    int size() const
    {
        int result = 0;
        for (const auto& sub_map : sub_map_vec_) {
            result += sub_map.size();
        }
        return result;
    }

    void addSubMap(const PointVector& new_points, const SE3d& pose)
    {
        // ROS_INFO("New submap added.");

        // submap
        sub_map_vec_.push_back(SubMap());

        sub_map_vec_.back().ikd_tree_->Build(new_points);
        sub_map_vec_.back().T_w_sub_ = pose;

        current_sub_map_id_++;

        // full map
        if (current_sub_map_id_ == 0) {
            full_map_.ikd_tree_->Build(new_points);
        }
    }

    void nearestSearch(PointType point, int k_nearest, PointVector& nearest_points, float max_dist = 100000)
    {
        std::vector<float> point_distance(k_nearest);
        // sub_map_vec_[current_sub_map_id_].ikd_tree_->Nearest_Search(point, k_nearest, nearest_points, point_distance, max_dist);
        
        full_map_.ikd_tree_->Nearest_Search(point, k_nearest, nearest_points, point_distance, max_dist);
    }

    void addPoints(PointVector& points_need_downsample, PointVector& points_no_need_downsample, bool key_frame=true)
    {
        // submap
        if (key_frame) {
            sub_map_vec_[current_sub_map_id_].ikd_tree_->Add_Points(points_need_downsample, true);
            sub_map_vec_[current_sub_map_id_].ikd_tree_->Add_Points(points_no_need_downsample, false);
        }

        // full map
        full_map_.ikd_tree_->Add_Points(points_need_downsample, true);
        full_map_.ikd_tree_->Add_Points(points_no_need_downsample, false);
    }

    Vector3d currentSubMapOrigin()
    {
        return sub_map_vec_[current_sub_map_id_].T_w_sub_.translation();
    }
};
