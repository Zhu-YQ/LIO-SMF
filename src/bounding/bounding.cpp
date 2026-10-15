#include "bounding.h"

// H = sum(d2_Ji / d2_d_zeta)
Matrix6d CalICPHessianWrtPose(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec)
{
    Matrix6d H = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, p_vec.size()),
        Matrix6d(Matrix6d::Zero()),
        [&](tbb::blocked_range<size_t> r, Matrix6d local_sum) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                const Vector3d& pi = p_vec[i];
                const Vector3d& ui = u_d_vec[i].block<3, 1>(0, 0);

                Eigen::Matrix<double, 1, 6> Ji;
                Ji.block<1, 3>(0, 0) = ui.transpose();
                Ji.block<1, 3>(0, 3) = -ui.transpose() * T.rotationMatrix() * SO3d::hat(pi);
                Matrix6d Hi = 2.0 * Ji.transpose() * Ji;
                local_sum += Hi;
            }
            return local_sum;
        },
        [](const Matrix6d& a, const Matrix6d& b) { return a + b; });

    return H;
}

// Hi = d_f / d_pi
std::vector<Matrix6x3d> CalICPDerivWrtSrcPt(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec, const Matrix6d& d2_J_d2_d_zeta_inv)
{
    std::vector<Matrix6x3d> Hi_vec(p_vec.size());

    tbb::parallel_for(size_t(0), p_vec.size(), [&](size_t i) {
        const Vector3d& pi = p_vec[i];
        const Vector3d& ui = u_d_vec[i].block<3, 1>(0, 0);
        const double di = u_d_vec[i](3);
        {
            const Eigen::Matrix<double, 1, 3> B = ui.transpose() * T.rotationMatrix();
            Hi_vec[i].block<3, 3>(0, 0) = ui * B;
            Hi_vec[i].block<3, 3>(3, 0) = SO3d::hat(pi) * B.transpose() * B;
            Hi_vec[i] *= 2.0;
            Hi_vec[i] = -d2_J_d2_d_zeta_inv * Hi_vec[i];
        }
    });

    return Hi_vec;
}

// Hi = d_f / d_ui
std::vector<Matrix6x3d> CalICPDerivWrtNormVec(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& u_d_vec, const Matrix6d& d2_J_d2_d_zeta_inv)
{
    std::vector<Matrix6x3d> Hi_vec(p_vec.size());

    tbb::parallel_for(size_t(0), p_vec.size(), [&](size_t i) {
        const Vector3d& pi = p_vec[i];
        const Vector3d T_pi = T * pi;
        const Matrix3d pi_hat = SO3d::hat(pi);

        const Vector3d& ui = u_d_vec[i].block<3, 1>(0, 0);
        const double di = u_d_vec[i](3);

        const Matrix3d R_T = T.rotationMatrix().transpose();

        {
            Hi_vec[i].block<3, 3>(0, 0) = ui.transpose() * T_pi * R_T + R_T * ui * T_pi.transpose() + di * R_T;
            Hi_vec[i].block<3, 3>(3, 0) = ui.transpose() * T_pi * pi_hat * R_T + pi_hat * R_T * ui * T_pi.transpose() + di * pi_hat * R_T;

            Hi_vec[i] = 2.0 * Hi_vec[i];
            Hi_vec[i] = -d2_J_d2_d_zeta_inv * Hi_vec[i];
        }
    });

    return Hi_vec;
}

// Hi = d_f / d_di
std::vector<Vector6d> CalICPDerivWrtD(const SE3d& T, const Vector3dVector& p_vec, const Vector4dVector& ud_vec, const Matrix6d& d2_J_d2_d_zeta_inv)
{
    std::vector<Vector6d> Hi_vec(p_vec.size());

    tbb::parallel_for(size_t(0), p_vec.size(), [&](size_t i) {
        const Vector3d& pi = p_vec[i];
        const Matrix3d pi_hat = SO3d::hat(pi);

        const Vector3d& ui = ud_vec[i].block<3, 1>(0, 0);
        const Matrix3d R_T = T.rotationMatrix().transpose();
        {
            Hi_vec[i].block<3, 1>(0, 0) = R_T * ui;
            Hi_vec[i].block<3, 1>(3, 0) = pi_hat * R_T * ui;
            Hi_vec[i] = 2.0 * Hi_vec[i];
            Hi_vec[i] = -d2_J_d2_d_zeta_inv * Hi_vec[i];
        }
    });

    return Hi_vec;
}
