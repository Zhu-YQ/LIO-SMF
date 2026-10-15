#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>

struct Interval {
    double inf_, sup_;

    Interval()
        : inf_(0)
        , sup_(0)
    {
    }

    Interval(double inf, double sup)
        : inf_(inf)
        , sup_(sup)
    {
    }

    Interval(const Interval& s)
        : inf_(s.inf_)
        , sup_(s.sup_)
    {
    }

    ~Interval() = default;

    bool isValid() const noexcept
    {
        return !(std::isnan(inf_) || std::isnan(sup_)) && inf_ <= sup_;
    }

    Interval& operator=(const Interval& s)
    {
        inf_ = s.inf_;
        sup_ = s.sup_;
        return *this;
    }

    Interval& operator=(Interval&& s) noexcept
    {
        inf_ = s.inf_;
        sup_ = s.sup_;
        return *this;
    }

    Interval operator+(const Interval& s) const
    {
        // [a, b] + [c, d] = [a + c, b + d]
        return { inf_ + s.inf_, sup_ + s.sup_ };
    }

    Interval operator-(const Interval& s) const
    {
        // [a, b] - [c, d] = [a - d, b - c]
        return { inf_ - s.sup_, sup_ - s.inf_ };
    }

    Interval operator*(const Interval& s) const
    {
        // [a, b] × [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
        double a = inf_;
        double b = sup_;
        double c = s.inf_;
        double d = s.sup_;
        return { std::min({ a * c, a * d, b * c, b * d }), std::max({ a * c, a * d, b * c, b * d }) };
    }

    Interval operator*(double k) const
    {
        return { inf_ * k, sup_ * k };
    }

    Interval intersect(const Interval& s) const
    {
        const double new_inf = std::max(inf_, s.inf_);
        const double new_sup = std::min(sup_, s.sup_);

        return Interval { new_inf, new_sup };
    }

    friend std::ostream& operator<<(std::ostream& os, const Interval& s)
    {
        os << "[" << s.inf_ << ", " << s.sup_ << "]";
        return os;
    }
};
