#pragma once

#include <Eigen/Core>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "interval.h"

#define PARALLELED

template <int N>
struct ElpSet {
    using VectorNd = Eigen::Matrix<double, N, 1>;
    using MatrixNd = Eigen::Matrix<double, N, N>;
    using MatrixXd = Eigen::MatrixXd;

    VectorNd a_;
    MatrixNd P_;

    ElpSet()
        : a_(VectorNd::Zero())
        , P_(0.5 * MatrixNd ::Identity())
    {
    }

    ElpSet(const VectorNd& a, const MatrixNd& P)
        : a_(a)
        , P_(P)
    {
    }

    ~ElpSet() = default;

    bool contain(const VectorNd& x) const
    {
        const auto dx = x - a_;
        return dx.transpose() * P_.inverse() * dx <= 1;
    }

    /* optimal Minkowski sum (plus) */
    ElpSet<N> operator+(const ElpSet<N>& e) const
    {
        // a
        const auto new_a = a_ + e.a_;

        // P
        auto obj_func = [this, e](double beta) {
            const auto new_P = (1.0 / (1.0 - beta)) * P_ + (1.0 / beta) * e.P_;
            return new_P.trace();
        };

        // line search
        double a = 1e-30, b = 1.0 - 1e-30;
        const double tau = (std::sqrt(5) + 1) / 2;
        double y = a + (b - a) / std::pow(tau, 2);
        double z = a + (b - a) / tau;
        while (b - a > 1e-5) {
            const double fy = obj_func(y);
            const double fz = obj_func(z);

            if (fy <= fz) {
                b = z;
                z = y;
                y = a + (b - a) / std::pow(tau, 2);
            }

            else {
                a = y;
                y = z;
                z = a + (b - a) / tau;
            }
        }
        const double beta = (a + b) / 2;
        const auto new_P = (1.0 / (1.0 - beta)) * P_ + (1.0 / beta) * e.P_;

        return { new_a, new_P };
    }

    /* optimal Minkowski sum (plus) with mutiple ellipsoidal sets*/
    static ElpSet<N> multiMinkSum(const std::vector<ElpSet<N>>& e_vec)
    {
#ifndef PARALLELED
        VectorNd new_a = VectorNd::Zero();
        MatrixNd new_P = MatrixNd::Zero();

        double sum_sqrt_trace = 0;
        for (const auto& e : e_vec) {
            sum_sqrt_trace += std::sqrt(e.P_.trace());
        }

        for (const auto& e : e_vec) {
            new_a += e.a_;
            new_P += (sum_sqrt_trace / std::sqrt(e.P_.trace())) * e.P_;
        }

        return { new_a, new_P };
#else
        double sum_sqrt_trace = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, e_vec.size()),
            0.0,
            [&](tbb::blocked_range<size_t> r, double local_sum) {
                for (size_t i = r.begin(); i != r.end(); i++) {
                    local_sum += std::sqrt(e_vec[i].P_.trace());
                }
                return local_sum;
            },
            [](double a, double b) { return a + b; });

        std::pair<VectorNd, MatrixNd> new_a_P_pair = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, e_vec.size()),
            std::pair<VectorNd, MatrixNd>(VectorNd::Zero(), MatrixNd::Zero()),
            [&](tbb::blocked_range<size_t> r, std::pair<VectorNd, MatrixNd> local_sum) {
                for (size_t i = r.begin(); i != r.end(); i++) {
                    local_sum.first += e_vec[i].a_;
                    local_sum.second += (sum_sqrt_trace / std::sqrt(e_vec[i].P_.trace())) * e_vec[i].P_;
                }
                return local_sum;
            },
            [](const std::pair<VectorNd, MatrixNd>& a, const std::pair<VectorNd, MatrixNd>& b) {
                return std::make_pair(a.first + b.first, a.second + b.second);
            });

        return { new_a_P_pair.first, new_a_P_pair.second };
#endif
    }

    /* optimal Minkowski sum (minus) */
    ElpSet<N> operator-(const ElpSet<N>& e) const
    {
        return (*this + (-1.0 * e));
    }

    /* scalar mutiplication */
    ElpSet<N> operator*(double k) const
    {
        return { k * a_, k * k * P_ };
    }

    friend ElpSet<N> operator*(double k, const ElpSet<N>& e)
    {

        return { k * e.a_, k * k * e.P_ };
    }

    template<int M>
    ElpSet<M> linearMap(const Eigen::Matrix<double, M, N>& A, const Eigen::Matrix<double, M, 1>& b = Eigen::Matrix<double, M, 1>::Zero()) const
    {   
        return { A * a_ + b, A * P_ * A.transpose() };
    }

    /* optimal intersection */
    ElpSet<N> intersect(const ElpSet<N>& e) const
    {
        auto obj_func = [this, e](double lamd) {
            const MatrixNd P1_inv = P_.inverse();
            const MatrixNd P2_inv = e.P_.inverse();

            const MatrixNd P_lamd_inv = lamd * P1_inv + (1.0 - lamd) * P2_inv;
            const MatrixNd P_lamd = P_lamd_inv.inverse();

            const VectorNd d_a = e.a_ - a_;
            double gamma = d_a.transpose() * P2_inv * P_lamd * P1_inv * d_a;
            gamma *= (lamd * (1.0 - lamd));
            // const MatrixNd P_inv = (1.0 / (1.0 - gamma)) * P_lamd_inv;
            const MatrixNd P = (1.0 - gamma) * P_lamd;

            return P.trace();
        };

        // line search
        double a = 1e-30, b = 1.0 - 1e-30;
        const double tau = (std::sqrt(5.0) + 1.0) / 2;
        double y = a + (b - a) / std::pow(tau, 2);
        double z = a + (b - a) / tau;
        while (b - a > 1e-5) {
            const double fy = obj_func(y);
            const double fz = obj_func(z);

            if (fy <= fz) {
                b = z;
                z = y;
                y = a + (b - a) / std::pow(tau, 2);
            }

            else {
                a = y;
                y = z;
                z = a + (b - a) / tau;
            }
        }
        const double lamd = (a + b) / 2;

        const MatrixNd P1_inv = P_.inverse();
        const MatrixNd P2_inv = e.P_.inverse();

        const MatrixNd P_lamd_inv = lamd * P1_inv + (1.0 - lamd) * P2_inv;
        const MatrixNd P_lamd = P_lamd_inv.inverse();

        const VectorNd d_a = e.a_ - a_;
        double gamma = d_a.transpose() * P2_inv * P_lamd * P1_inv * d_a;
        gamma *= (lamd * (1.0 - lamd));
        // const auto P_inv = (1.0 / (1.0 - gamma)) * P_lamd_inv;

        return { P_lamd * (lamd * P1_inv * a_ + (1.0 - lamd) * P2_inv * e.a_), (1.0 - gamma) * P_lamd };
    }

    bool isnan() const
    {
        return std::isnan(a_.sum() + P_.sum());
    }

    Eigen::Matrix<Interval, N, 1> toBox() const
    {
        Eigen::Matrix<Interval, N, 1> interval;
        for (int i = 0; i < N; i++) {
            const double radius = std::sqrt(P_(i, i));
            interval(i) = Interval(a_(i) - radius, a_(i) + radius);
        }

        return interval;
    }

    static ElpSet<N> FromBox(const Eigen::Matrix<Interval, N, 1>& box)
    {
        ElpSet<N> e(VectorNd::Zero(), MatrixNd::Zero());
        for (int i = 0; i < N; i++) {
            const double center = 0.5 * (box(i).sup_ + box(i).inf_);
            const double radius = box(i).sup_ - center;

            e.a_(i) = center;
            e.P_(i, i) = double(N) * radius * radius;
        }

        return e;
    }
};
