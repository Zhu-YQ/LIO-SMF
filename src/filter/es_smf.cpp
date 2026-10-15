#include "es_smf.h"

void ESSMF::setBias(const Vector3d& ba, const Matrix3d& N_ba, const Vector3d& bg, const Matrix3d& N_bg)
{
    ba_ = ba;
    bg_ = bg;

    d_ba_.P_ = N_ba;
    d_bg_.P_ = N_bg;
}

void ESSMF::setGravity(const Vector3d& g)
{
    g_ = g;
}

void ESSMF::setTime(double time_stamp)
{
    time_stamp_ = time_stamp;
}

void ESSMF::alignGravity()
{
    Vector3d ez(0, 0, -1), gz(g_);
    Eigen::Quaterniond G_q_I0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
    G_q_I0.normalize();
    SO3d G_R_I0(G_q_I0);

    t_ = G_R_I0 * t_;
    R_ = G_R_I0 * R_;
    v_ = G_R_I0 * v_;
    g_ = G_R_I0 * g_;
}

Vector3d ESSMF::getPosition()
{
    return t_;
}

Quatd ESSMF::getOrientation()
{
    return R_.unit_quaternion();
}

nav_msgs::Odometry ESSMF::getPose()
{
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time(time_stamp_);
    odom.header.frame_id = "loc_init";
    odom.child_frame_id = "body";

    odom.pose.pose.position.x = t_[0];
    odom.pose.pose.position.y = t_[1];
    odom.pose.pose.position.z = t_[2];

    const auto q = getOrientation();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();

    Eigen::Matrix<double, 6, 6> full_shape_mat = Eigen::Matrix<double, 6, 6>::Zero();

    // local
    // full_shape_mat.block<3, 3>(0, 0) = d_t_.P_;
    // full_shape_mat.block<3, 3>(3, 3) = d_theta_.P_;

    // global
    SE3d& T_g_w = T_map_anchor_;
    SE3d T_g_i(R_, t_);
    SE3d T_w_i = T_g_w.inverse() * T_g_i;

    vector<ES3> d_t_global_compose_vec;
    d_t_global_compose_vec.push_back(d_t_map_);
    d_t_global_compose_vec.push_back(d_t_.linearMap(T_g_w.rotationMatrix()));
    if (T_w_i.translation().norm() != 0) {
        d_t_global_compose_vec.push_back(d_theta_map_.linearMap(static_cast<Matrix3d>(-T_g_w.rotationMatrix() * SO3d::hat(T_w_i.translation()))));
    }
    d_t_global_compose_vec.emplace_back(Vector3d::Zero(), global_uncertainty_compensation_t_ * Matrix3d::Identity());
    d_t_global_ = ES3::multiMinkSum(d_t_global_compose_vec);

    vector<ES3> d_theta_global_compose_vec;
    d_theta_global_compose_vec.push_back(d_theta_map_.linearMap(static_cast<Matrix3d>(T_w_i.rotationMatrix().transpose())));
    d_theta_global_compose_vec.push_back(d_theta_);
    d_theta_global_compose_vec.emplace_back(Vector3d::Zero(), global_uncertainty_compensation_r_ * Matrix3d::Identity());
    d_theta_global_ = ES3::multiMinkSum(d_theta_global_compose_vec);

    full_shape_mat.block<3, 3>(0, 0) = d_t_global_.P_;
    full_shape_mat.block<3, 3>(3, 3) = d_theta_global_.P_;

    double* full_shape_mat_data = full_shape_mat.data();
    for (size_t i = 0; i < 36; i++) {
        odom.pose.covariance[i] = full_shape_mat_data[i];
    }

    return odom;
}

sensor_msgs::PointCloud2::ConstPtr ESSMF::getMapPointCloud()
{
    sensor_msgs::PointCloud2::Ptr map_point_cloud_pc2(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(visual_map_, *map_point_cloud_pc2);
    map_point_cloud_pc2->header.frame_id = "loc_init";
    map_point_cloud_pc2->header.stamp = ros::Time::now();

    return map_point_cloud_pc2;
}

void ESSMF::predict(const sensor_msgs::Imu::ConstPtr& imu_msg, const Matrix3d& N_a, const Matrix3d& N_g)
{
    if (imu_msg->header.stamp.toSec() < time_stamp_) {
        return;
    }

    const double dt = imu_msg->header.stamp.toSec() - time_stamp_;
    time_stamp_ = imu_msg->header.stamp.toSec();

    Vector3d a(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    a -= ba_;

    Vector3d w(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    w -= bg_;

    /* error state */
    // noises of IMU
    const ES3 n_a(Vector3d::Zero(), N_a);
    const ES3 n_g(Vector3d::Zero(), N_g);

    // d_t, d_v, d_theta are expanded
    // d_ba, d_bg are unchanged
    d_t_ = d_t_ + (dt * d_v_);

    const Matrix3d A = -R_.matrix() * SO3d::hat(a) * dt;
    const Matrix3d B = -dt * R_.matrix();
    d_v_ = ES3::multiMinkSum({ d_v_, d_theta_.linearMap<3>(A), d_ba_.linearMap<3>(B), dt * n_a });

    const Matrix3d C = SO3d::exp(-w * dt).matrix();
    d_theta_ = ES3::multiMinkSum({ d_theta_.linearMap<3>(C), dt * d_bg_, dt * n_g });

    /* nominal state */
    // basic method from FAST-LIO
    t_ = t_ + v_ * dt + 0.5 * (R_ * a) * dt * dt + 0.5 * g_ * dt * dt;
    v_ = v_ + R_ * a * dt + g_ * dt;
    R_ = R_ * SO3d::exp(w * dt);

    // high-order method from LIO-EKF
    // const Vector3d v_last = v_;
    // v_ = v_last + g_ * dt + R_ * (a * dt + 0.5 * w.cross(a) * dt * dt + (1.0 / 12) * (w_last_.cross(a) - w.cross(a_last_)) * dt * dt);
    // t_ = t_ + 0.5 * (v_ + v_last) * dt;
    // R_ = R_ * SO3d::exp(w * dt + (1.0 / 12) * w_last_.cross(w) * dt * dt);

    // a_last_ = a;
    // w_last_ = w;
}

const int NUM_MATCH_POINTS = 5;

template <typename T>
inline bool esti_plane(Eigen::Matrix<T, 4, 1>& pca_result, const PointVector& point, const T& threshold)
{
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }

    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (std::fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold) {
            return false;
        }
    }
    return true;
}

std::vector<Vector3d> PCLToEigen(const PointCloudType::ConstPtr& point_cloud)
{
    std::vector<Vector3d> point_vec(point_cloud->size());
    tbb::parallel_for(size_t(0), point_cloud->size(), [&](size_t i) {
        const auto& p_pcl = point_cloud->points[i];
        point_vec[i] = Vector3d(p_pcl.x, p_pcl.y, p_pcl.z);
    });

    return point_vec;
}

ES3 ESSMF::calPointNoiseES(const Vector3d& p_imu)
{
    const Vector3d p_lidar = SE3d(R_imu_lidar_, t_imu_lidar_).inverse() * p_imu;
    const Vector3d phi = p_lidar.normalized();
    const double d = p_lidar.norm();

    // N
    const Vector3d a1 = p_lidar;

    Vector3d a2 = Vector3d::Zero();
    if (std::fabs(a1(0)) > 1e-3) {
        a2(1) = 1;
    } else {
        a2(0) = 1;
    }

    const Vector3d a3 = a1.cross(a2);

    // schmidt
    const Vector3d& b1 = a1;
    const Vector3d b2 = a2 - (a2.dot(b1) / b1.dot(b1)) * b1;
    const Vector3d b3 = a3 - (a3.dot(b1) / b1.dot(b1)) * b1 - (a3.dot(b2) / b2.dot(b2)) * b2;

    // const Vector3d n1 = b1.normalized();
    const Vector3d n2 = b2.normalized();
    const Vector3d n3 = b3.normalized();

    Eigen::Matrix<double, 3, 2> N = Eigen::Matrix<double, 3, 2>::Zero();
    N.block<3, 1>(0, 0) = n2;
    N.block<3, 1>(0, 1) = n3;

    Matrix3d A;
    A.block<3, 1>(0, 0) = phi;
    A.block<3, 2>(0, 1) = -d * SO3d::hat(phi) * N;

    Matrix3d P_p = Matrix3d::Zero();
    P_p(0, 0) = 3.0 * std::pow(range_bound_, 2);
    P_p(1, 1) = 3.0 * std::pow(M_PI * bearing_bound_ / 180.0, 2);
    P_p(2, 2) = 3.0 * std::pow(M_PI * bearing_bound_ / 180.0, 2);

    ES3 p_es(Vector3d::Zero(), P_p);
    p_es = p_es.linearMap<3>(A, Vector3d::Zero());
    p_es = p_es.linearMap<3>(R_imu_lidar_.matrix(), Vector3d::Zero());
    p_es = p_es + ES3(Vector3d::Zero(), point_compensation_ * Matrix3d::Identity());

    return p_es;
}

// inline double square(double x) { return x * x; }

void ESSMF::updateErrorState(const PointCloudType::ConstPtr& current_point_cloud, double lidar_time_stamp)
{
    if (current_point_cloud->empty()) {
        ROS_WARN("No points.");
        return;
    }

    if (current_point_cloud->size() < 10) {
        valid_update_ = false;
        return;
    }

    if (map_.empty()) {
        map_.addSubMap(current_point_cloud->points, SE3d());
    }

    time_stamp_ = lidar_time_stamp;

    /* ICP */
    valid_update_ = true;

    const auto source_body = PCLToEigen(current_point_cloud);

    R_icp_ = R_;
    t_icp_ = t_;

    // degeneracy
    bool degeneracy_detection_done = false;

    Vector3d omg_vec_r = Vector3d::Zero();
    Vector3d omg_vec_t = Vector3d::Zero();

    Matrix3d V_r;
    Matrix3d V_t;

    const auto icp_start_time = std::chrono::high_resolution_clock::now();
    for (int iter_num = 0; iter_num < max_iteration_; iter_num++) {
        std::mutex icp_mutex;

        int i_save = -1; // assign only id for valid measurement

        icp_source_vec_.resize(source_body.size());
        icp_plane_ud_vec_.resize(source_body.size());
        // icp_plane_pts_vec_.resize(source_body.size());
        icp_point_error_es_.resize(source_body.size());

        icp_distances_vec_.resize(source_body.size());
        icp_J_vec_.resize(source_body.size());

        tbb::parallel_for(size_t(0), source_body.size(), [&](size_t i) {
            // if (source_body[i].norm() >= 1000) {
            //     return;
            // }

            const Vector3d p_world_eigen = R_icp_ * source_body[i] + t_icp_;

            PointType p_world_pcl;
            p_world_pcl.x = p_world_eigen(0);
            p_world_pcl.y = p_world_eigen(1);
            p_world_pcl.z = p_world_eigen(2);

            // find point
            PointVector search_result;
            map_.nearestSearch(p_world_pcl, NUM_MATCH_POINTS, search_result);

            if (search_result.size() < NUM_MATCH_POINTS) {
                return;
            }

            // fit plane
            Vector4d plane_abcd;
            if (!esti_plane(plane_abcd, search_result, 0.1)) {
                return;
            }

            const Vector3d u(plane_abcd(0), plane_abcd(1), plane_abcd(2));
            const double d = plane_abcd(3);
            const double distance = u.transpose() * p_world_eigen + d;

            const double s = 1.0 - 0.9 * std::abs(distance) / std::sqrt(source_body[i].norm());

            if (s <= 0.9) {
                return;
            }

            // J
            Eigen::Matrix<double, 1, 6> J_r;
            J_r.block<1, 3>(0, 0) = u.transpose();
            J_r.block<1, 3>(0, 3) = -u.transpose() * R_icp_.matrix() * SO3d::hat(source_body[i]);

            // uncertainty model for source point
            const ES3 p_error_es = calPointNoiseES(source_body[i]);

            int current_i_save = __sync_add_and_fetch(&i_save, 1);

            icp_source_vec_[current_i_save] = source_body[i];
            icp_plane_ud_vec_[current_i_save] = plane_abcd;
            // icp_plane_pts_vec_[current_i_save] = search_result;
            icp_point_error_es_[current_i_save] = p_error_es;

            icp_distances_vec_[current_i_save] = distance;
            icp_J_vec_[current_i_save] = J_r;
        });

        icp_source_vec_.resize(i_save + 1);
        icp_plane_ud_vec_.resize(i_save + 1);
        // icp_plane_pts_vec_.resize(i_save + 1);
        icp_point_error_es_.resize(i_save + 1);

        icp_distances_vec_.resize(i_save + 1);
        icp_J_vec_.resize(i_save + 1);

        const auto& [J, JTJ, JTr] = tbb::parallel_reduce(
            tbb::blocked_range<size_t> { 0, icp_source_vec_.size() },
            ResultTuple(),
            [&](const tbb::blocked_range<size_t>& range, ResultTuple J_temp) -> ResultTuple {
                auto& [J_private, JTJ_private, JTr_private] = J_temp;
                for (auto i = range.begin(); i < range.end(); i++) {

                    const auto& J_r = icp_J_vec_[i];
                    const double residual = icp_distances_vec_[i];

                    // auto Weight = [&](double residual2) {
                    //     const double delta_2 = square(0.2);
                    //     return (2.0 * square(delta_2)) / square(delta_2 + residual2);
                    // };
                    // const double w = Weight(residual * residual);

                    const double w = 1;

                    J_private += J_r;
                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;
                }
                return J_temp;
            },
            [&](ResultTuple a, const ResultTuple& b) -> ResultTuple { return a + b; });

        /* degeneracy detection */
        if (deg_process_en_ && !degeneracy_detection_done) {
            Matrix6d H_icp = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, icp_source_vec_.size()),
                Matrix6d(Matrix6d::Zero()),
                [&](tbb::blocked_range<size_t> r, Matrix6d local_sum) {
                    for (size_t i = r.begin(); i != r.end(); i++) {
                        const Vector3d& pi = icp_source_vec_[i];
                        const Vector3d& ui = icp_plane_ud_vec_[i].block<3, 1>(0, 0);

                        Eigen::Matrix<double, 1, 6> J;
                        J.block<1, 3>(0, 0) = ui.transpose();
                        J.block<1, 3>(0, 3) = -ui.transpose() * R_icp_.matrix() * SO3d::hat(pi);
                        local_sum += J.transpose() * J;
                    }
                    return local_sum;
                },
                [](const Matrix6d& a, const Matrix6d& b) { return a + b; });
            
            // translation
            const Matrix3d H_icp_t = H_icp.block<3, 3>(0, 0);
            
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver_t(H_icp_t);
            Vector3d eigenvalues_t = eigensolver_t.eigenvalues();
            Vector3d normalized_eigenvalues_t = (1.0 / (eigenvalues_t[2] + 0.001)) * eigenvalues_t;
            V_t = eigensolver_t.eigenvectors();

            for (int k = 0; k < 3; k++) {
                if (eigenvalues_t[k] < k_abs_t_ || normalized_eigenvalues_t[k] < k_rel_t_) {
                    omg_vec_t[k] = 1;
                }
            }
            cout << eigenvalues_t.transpose() << "\n";

            degeneracy_t_ = (omg_vec_t.sum() != 0);

            // rotation
            const Matrix3d H_icp_r = H_icp.block<3, 3>(3, 3);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver_r(H_icp_r);
            Vector3d eigenvalues_r = eigensolver_r.eigenvalues();
            Vector3d normalized_eigenvalues_r = (1.0 / (eigenvalues_r[2] + 0.001)) * eigenvalues_r;
            V_r = eigensolver_r.eigenvectors();

            for (int k = 0; k < 3; k++) {
                if (eigenvalues_r[k] < k_abs_r_ || normalized_eigenvalues_r[k] < k_rel_r_) {
                    omg_vec_r[k] = 1;
                }
            }
            cout << eigenvalues_r.transpose() << "\n\n";

            degeneracy_r_ = (omg_vec_r.sum() != 0);

            degeneracy_detection_done = true;

            if (degeneracy_t_ || degeneracy_r_) {
                ROS_WARN("Degeneracy detected!");
            }
        }

        /* G-N method */
        Vector6d d_zeta = JTJ.ldlt().solve(-JTr);
        Vector3d d_zeta_t = d_zeta.block<3, 1>(0, 0);
        Vector3d d_zeta_r = d_zeta.block<3, 1>(3, 0);

        if (deg_process_en_) {
            if (degeneracy_t_) {
                Matrix3d M = Matrix3d::Zero();
                M(0, 0) = !omg_vec_t[0];
                M(1, 1) = !omg_vec_t[1];
                M(2, 2) = !omg_vec_t[2];

                d_zeta_t = V_t * M * V_t.transpose() * d_zeta_t;
            }

            if (degeneracy_r_) {
                Matrix3d M = Matrix3d::Zero();
                M(0, 0) = !omg_vec_r[0];
                M(1, 1) = !omg_vec_r[1];
                M(2, 2) = !omg_vec_r[2];

                d_zeta_r = V_r * M * V_r.transpose() * d_zeta_r;
            }
        }

        R_icp_ = R_icp_ * SO3d::exp(d_zeta_r);
        t_icp_ = t_icp_ + d_zeta_t;

        d_zeta.block<3, 1>(0, 0) = d_zeta_t ;
        d_zeta.block<3, 1>(3, 0) = d_zeta_r;
        if (d_zeta.norm() < 0.001) {
            break;   
        }
    }

    const auto icp_end_time = std::chrono::high_resolution_clock::now();
    icp_duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(icp_end_time - icp_start_time).count() * 10e-9;

    if (icp_distances_vec_.size() < 10) {
        valid_update_ = false;
        ROS_WARN("Invalid ICP! LIO is incapacitated!");
        ros::shutdown();
        return;
    }

    /* uncertainty */
    const auto uncertainty_start_time = std::chrono::high_resolution_clock::now();

    // point part 1
    const Matrix6d d2_J_d2_dzeta = CalICPHessianWrtPose(SE3d(R_icp_, t_icp_), icp_source_vec_, icp_plane_ud_vec_);
    const Matrix6d d2_J_d2_dzeta_inv = d2_J_d2_dzeta.inverse();
    const std::vector<Matrix6x3d> df_dp_vec = CalICPDerivWrtSrcPt(SE3d(R_icp_, t_icp_), icp_source_vec_, icp_plane_ud_vec_, d2_J_d2_dzeta_inv);

    std::vector<ES6> es_vec;
    es_vec.resize(df_dp_vec.size() * 2.0);

    tbb::parallel_for(size_t(0), df_dp_vec.size(), [&](size_t i) {
        // point part 2
        const ES3& p_error_es = icp_point_error_es_[i];

        es_vec[i * 2.0] = p_error_es.linearMap<6>(df_dp_vec[i]);
        es_vec[i * 2.0 + 1] = ES6(Vector6d::Zero(), 1e-10 * Matrix6d::Identity());
    });

    // for wrong convergence and poor initialization
    Vector6d vec_lump;
    vec_lump << lumped_trans_, lumped_trans_, lumped_trans_, lumped_rot_, lumped_rot_, lumped_rot_;
    es_vec.emplace_back(Vector6d::Zero(), vec_lump.asDiagonal());

    ES6 d_zeta = ES6::multiMinkSum(es_vec);

    const auto uncertainty_end_time = std::chrono::high_resolution_clock::now();
    uncertainty_duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(uncertainty_end_time - uncertainty_start_time).count() * 10e-9;

    const auto intersection_start_time = std::chrono::high_resolution_clock::now();

    // t
    ES3 d_t_observed;
    Matrix3x6d A_t = Matrix3x6d::Zero();
    A_t.block<3, 3>(0, 0) = Matrix3d::Identity();
    d_t_observed = d_zeta.linearMap<3>(A_t, Vector3d::Zero());
    d_t_observed.a_ = t_icp_ - t_;

    if (!degeneracy_t_) {
        d_t_ = d_t_.intersect(d_t_observed);
    } else {
        /* degeneracy process */
        const ES3 d_t_V = d_t_.linearMap<3>(V_t.transpose());
        const ES3 d_t_observed_V = d_t_observed.linearMap<3>(V_t.transpose());

        Eigen::Matrix<Interval, 3, 1> d_t_box = d_t_V.toBox();
        Eigen::Matrix<Interval, 3, 1> d_t_observed_box = d_t_observed_V.toBox();
        Eigen::Matrix<Interval, 3, 1> d_t_updated_box;
        for (int i = 0; i < 3; i++) {
            if (omg_vec_t(i) == 0) {
                d_t_updated_box(i) = d_t_box(i).intersect(d_t_observed_box(i));
                if (d_t_updated_box(i).inf_ > d_t_updated_box(i).sup_) {
                    d_t_updated_box(i) = d_t_box(i);
                }
            } else {
                d_t_updated_box(i) = d_t_box(i);
            }
        }
        // d_t_ = ES3::FromBox(d_t_updated_box).linearMap<3>(V_t);

        ES3 d_t_temp = ES3::FromBox(d_t_updated_box).linearMap<3>(V_t);
        d_t_ = d_t_.intersect(d_t_temp);
    }

    // v
    ES3 d_v_observed(Vector3d::Zero(), 0.01 * Matrix3d::Identity());
    const ES3 t_es_observed = ES3(t_icp_, d_t_observed.P_);
    if (t_es_observed_last_time_stamp_ != -1) {
        const double dt = lidar_time_stamp - t_es_observed_last_time_stamp_;
        d_v_observed = t_es_observed - t_es_observed_last_;
        d_v_observed = (1.0 / dt) * d_v_observed;
        d_v_observed.a_ -= v_;
    }

    t_es_observed_last_ = t_es_observed;
    t_es_observed_last_time_stamp_ = lidar_time_stamp;

    if (!degeneracy_t_) {
        d_v_ = d_v_.intersect(d_v_observed);
    } else {
        /* degeneracy process */
        const ES3 d_v_V = d_v_.linearMap<3>(V_t.transpose());
        const ES3 d_v_observed_V = d_v_observed.linearMap<3>(V_t.transpose());

        Eigen::Matrix<Interval, 3, 1> d_v_box = d_v_V.toBox();
        Eigen::Matrix<Interval, 3, 1> d_v_observed_box = d_v_observed_V.toBox();
        Eigen::Matrix<Interval, 3, 1> d_v_updated_box;
        for (int k = 0; k < 3; k++) {
            if (omg_vec_t(k) == 0) {
                d_v_updated_box(k) = d_v_box(k).intersect(d_v_observed_box(k));

                if (d_v_updated_box(k).inf_ > d_v_updated_box(k).sup_) {
                    d_v_updated_box(k) = d_v_box(k);
                }
            } else {
                d_v_updated_box(k) = d_v_box(k);
            }
        }
        // d_v_ = ES3::FromBox(d_v_updated_box).linearMap<3>(V_t);

        const ES3 d_v_temp = ES3::FromBox(d_v_updated_box).linearMap<3>(V_t);
        d_v_ = d_v_.intersect(d_v_temp);
    }

    // theta
    ES3 d_theta_observed;
    Matrix3x6d A_theta = Matrix3x6d::Zero();
    A_theta.block<3, 3>(0, 3) = Matrix3d::Identity();
    d_theta_observed = d_zeta.linearMap<3>(A_theta);

    const auto J_inv = SO3d::leftJacobianInverse(-(R_.inverse() * R_icp_).log());
    d_theta_observed = d_theta_observed.linearMap<3>(J_inv);
    d_theta_observed.a_ = (R_.inverse() * R_icp_).log();
    d_theta_observed = d_theta_observed + ES3(Vector3d::Zero(), bch_compensation_ * Matrix3d::Identity());

    if (!degeneracy_r_) {
        d_theta_ = d_theta_.intersect(d_theta_observed);
    } else {
        /* degeneracy process */
        const ES3 d_theta_V = d_theta_.linearMap<3>(V_r.transpose());
        const ES3 d_theta_observed_V = d_theta_observed.linearMap<3>(V_r.transpose());

        Eigen::Matrix<Interval, 3, 1> d_theta_box = d_theta_V.toBox();
        Eigen::Matrix<Interval, 3, 1> d_theta_observed_box = d_theta_observed_V.toBox();
        Eigen::Matrix<Interval, 3, 1> d_theta_updated_box;
        for (int k = 0; k < 3; k++) {
            if (omg_vec_r(k) == 0) {
                d_theta_updated_box(k) = d_theta_box(k).intersect(d_theta_observed_box(k));

                if (d_theta_updated_box(k).inf_ > d_theta_updated_box(k).sup_) {
                    d_theta_updated_box(k) = d_theta_box(k);
                }
            } else {
                d_theta_updated_box(k) = d_theta_box(k);
            }
        }
        // d_theta_ = ES3::FromInterval(d_theta_updated_interval).affineTransform<3>(V_r);

        const ES3 d_theta_temp = ES3::FromBox(d_theta_updated_box).linearMap<3>(V_r);
        d_theta_ = d_theta_.intersect(d_theta_temp);
    }

    const auto intersection_end_time = std::chrono::high_resolution_clock::now();
    intersection_duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(intersection_end_time - intersection_start_time).count() * 10e-9;
}

void ESSMF::updateNominalState()
{
    if (!valid_update_) {
        return;
    }

    t_ += d_t_.a_;
    d_t_.a_ = Vector3d::Zero();

    v_ += d_v_.a_;
    d_v_.a_ = Vector3d::Zero();

    R_ = R_ * SO3d::exp(d_theta_.a_);
    d_theta_.a_ = Vector3d::Zero();
}

bool ESSMF::zeroVelocityUpdate(double velocity_bound)
{
    if (!std::isfinite(velocity_bound) || velocity_bound <= 0.0) {
        return false;
    }

    const Matrix3d P_zupt = 3.0 * velocity_bound * velocity_bound * Matrix3d::Identity();
    const ES3 d_v_zupt(-v_, P_zupt);
    const ES3 d_v_updated = d_v_.intersect(d_v_zupt);

    const Matrix3d P_symmetric = 0.5 * (d_v_updated.P_ + d_v_updated.P_.transpose());
    if (!d_v_updated.a_.allFinite() || !P_symmetric.allFinite()) {
        return false;
    }

    Eigen::SelfAdjointEigenSolver<Matrix3d> eigensolver(P_symmetric);
    if (eigensolver.info() != Eigen::Success || eigensolver.eigenvalues().minCoeff() <= 0.0) {
        return false;
    }

    d_v_ = ES3(d_v_updated.a_, P_symmetric);
    return true;
}

template <typename PointType>
float calc_dist(PointType a, PointType b)
{
    float dist = 0.0f;
    dist = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    return dist;
}

void ESSMF::updateMap(const PointCloudType::ConstPtr& current_point_cloud)
{
    if (!valid_update_ || degeneracy_t_ || degeneracy_r_) {
        return;
    }

    if (icp_distances_vec_.empty()) {
        return;
    }

    /* key frame determination */
    bool key_frame = false;
    if ((t_ - t_last_key_frame_).norm() > 2
        || (R_.inverse() * R_last_key_frame_).log().norm() > (M_PI / 180.0) * 40) {
        key_frame = true;

        R_last_key_frame_ = R_;
        t_last_key_frame_ = t_;
    }

    /* degeneracy process */
    double filter_size_map_min = 0.2;

    // std::mutex vector_mutex;

    PointVector points_need_downsample;
    int num_points_need_downsample = 0;

    PointVector points_no_need_downsample;
    int num_points_no_need_downsample = 0;

    // PointVector points_all;

    points_need_downsample.resize(current_point_cloud->size());
    points_no_need_downsample.resize(current_point_cloud->size());

    tbb::parallel_for(size_t(0), current_point_cloud->size(), [&](size_t i) {
        /* transform to world frame */
        const PointType& pi_body_pcl = current_point_cloud->points[i];
        const Vector3d pi_body_eigen(pi_body_pcl.x, pi_body_pcl.y, pi_body_pcl.z);

        // if (pi_body_eigen.norm() >= 1000) {
        //     return;
        // }

        const Vector3d pi_world_eigen = R_ * pi_body_eigen + t_;
        // const Vector3d pi_world_eigen = R_icp_ * pi_body_eigen + t_icp_;

        PointType pi_world_pcl;
        pi_world_pcl.x = pi_world_eigen(0);
        pi_world_pcl.y = pi_world_eigen(1);
        pi_world_pcl.z = pi_world_eigen(2);

        /* add all */
        // {
        //     std::unique_lock<std::mutex> lock(vector_mutex);
        //     points_all.push_back(pi_world_pcl);
        // }

        /* decide if need add to map */
        PointVector search_result;

        map_.nearestSearch(pi_world_pcl, NUM_MATCH_POINTS, search_result);

        if (!search_result.empty()) {
            bool need_add = true;
            PointType mid_point;
            mid_point.x = std::floor(pi_world_pcl.x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = std::floor(pi_world_pcl.y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = std::floor(pi_world_pcl.z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(pi_world_pcl, mid_point);
            if (std::fabs(search_result[0].x - mid_point.x) > 0.5 * filter_size_map_min
                && std::fabs(search_result[0].y - mid_point.y) > 0.5 * filter_size_map_min
                && std::fabs(search_result[0].z - mid_point.z) > 0.5 * filter_size_map_min) {

                int target_i = __sync_fetch_and_add(&num_points_no_need_downsample, 1);
                points_no_need_downsample[target_i] = pi_world_pcl;

                // points_no_need_downsample.push_back(pi_world_pcl);

                return;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                if (search_result.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(search_result[readd_i], mid_point) < dist) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) {
                // std::unique_lock<std::mutex> lock(vector_mutex);
                // points_need_downsample.push_back(pi_world_pcl);

                int target_i = __sync_fetch_and_add(&num_points_need_downsample, 1);
                points_need_downsample[target_i] = pi_world_pcl;
            }
        } else {
            // std::unique_lock<std::mutex> lock(vector_mutex);
            // points_need_downsample.push_back(pi_world_pcl);

            int target_i = __sync_fetch_and_add(&num_points_need_downsample, 1);
            points_need_downsample[target_i] = pi_world_pcl;
        }
    });

    points_need_downsample.resize(num_points_need_downsample);
    points_no_need_downsample.resize(num_points_no_need_downsample);

    if ((t_ - map_.currentSubMapOrigin()).norm() > local_map_distance_) {
        // add new submap
        map_.addSubMap(points_no_need_downsample, SE3d(R_, t_));
        map_.addPoints(points_need_downsample, points_no_need_downsample);

        // propagate map uncertainty
        SO3d R_g_wp = R_;
        Vector3d t_g_wp = t_;
        SE3d T_g_wp(R_g_wp, t_g_wp);

        SE3d& T_g_w = T_map_anchor_;
        SE3d T_w_wp = T_g_w.inverse() * T_g_wp;

        ES3 d_t_loc_swc = d_t_;
        ES3 d_theta_loc_swc = d_theta_;

        vector<ES3> new_d_t_compose_vec;
        new_d_t_compose_vec.push_back(d_t_map_);
        new_d_t_compose_vec.push_back(d_t_loc_swc.linearMap(T_g_w.rotationMatrix()));
        if (T_w_wp.translation().norm() != 0) {
            new_d_t_compose_vec.push_back(d_theta_loc_swc.linearMap(static_cast<Matrix3d>(-T_g_w.rotationMatrix() * SO3d::hat(T_w_wp.translation()))));
        }
        new_d_t_compose_vec.emplace_back(Vector3d::Zero(), map_anchor_compensation_t_ * Matrix3d::Identity());
        d_t_map_ = ES3::multiMinkSum(new_d_t_compose_vec);

        vector<ES3> new_d_theta_compose_vec;
        new_d_theta_compose_vec.push_back(d_theta_map_.linearMap(static_cast<Matrix3d>(T_w_wp.rotationMatrix().transpose())));
        new_d_theta_compose_vec.push_back(d_theta_loc_swc);
        new_d_theta_compose_vec.emplace_back(Vector3d::Zero(), map_anchor_compensation_r_ * Matrix3d::Identity());
        d_theta_map_ = ES3::multiMinkSum(new_d_theta_compose_vec);

        T_map_anchor_ = T_g_wp;

    } else {
        map_.addPoints(points_need_downsample, points_no_need_downsample, key_frame);
    }

    /* simple map for visualization */
    // visual_map_.points.insert(visual_map_.points.end(), points_need_downsample.begin(), points_need_downsample.end());
    // pcl::VoxelGrid<PointType> vg;
    // vg.setInputCloud(visual_map_.makeShared());
    // // vg.setLeafSize(0.5f, 0.5f, 0.5f);
    // vg.setLeafSize(0.2f, 0.2f, 0.2f); // for better visualization
    // vg.filter(visual_map_);
}
