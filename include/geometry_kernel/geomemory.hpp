// Geometry Kernel / GeoMemory core, C++17.
//
// Header-only implementation of the GeoMemory primitives from the document:
// Poincare-ball arithmetic, tangent-space user memory updates, scoring, and
// ternary recommendation loss. The implementation is intentionally dependency
// free and uses RAII-only storage, so there are no manual ownership paths.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace geometry_kernel {

constexpr double kNormEps = 1e-15;
constexpr double kDenomEps = 1e-15;
constexpr double kBallEps = 1e-9;

// ---------- scalar helpers ----------

inline double finite_or_zero(double x) noexcept {
    return std::isfinite(x) ? x : 0.0;
}

inline double dot(const double* x, const double* y, std::size_t d) noexcept {
    double s = 0.0;
    for (std::size_t k = 0; k < d; ++k) {
        s += finite_or_zero(x[k]) * finite_or_zero(y[k]);
    }
    return s;
}

inline double squared_norm(const double* x, std::size_t d) noexcept {
    return dot(x, x, d);
}

inline double norm(const double* x, std::size_t d) noexcept {
    double s = 0.0;
    for (std::size_t k = 0; k < d; ++k) {
        s = std::hypot(s, finite_or_zero(x[k]));
    }
    return s;
}

inline double sigmoid(double x) noexcept {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

inline double log_sigmoid(double x) noexcept {
    if (x >= 0.0) return -std::log1p(std::exp(-x));
    return x - std::log1p(std::exp(x));
}

inline double ball_max_norm(double c, double eps = kBallEps) noexcept {
    return (c <= 0.0) ? 0.0 : (1.0 - eps) / std::sqrt(c);
}

inline double projection_scale_from_norm(double n, double c,
                                         double eps = kBallEps) noexcept {
    if (c <= 0.0) return 1.0;
    if (!std::isfinite(n)) return 0.0;
    const double max_n = ball_max_norm(c, eps);
    return (n > max_n && n > 0.0) ? (max_n / n) : 1.0;
}

inline double clamp_unit_open(double x, double eps = kBallEps) noexcept {
    if (!std::isfinite(x)) return 1.0 - eps;
    if (x < 0.0) return 0.0;
    if (x > 1.0 - eps) return 1.0 - eps;
    return x;
}

// Project a vector into the open Poincare ball. Aliasing out == x is allowed.
inline void project_to_ball(const double* x, std::size_t d,
                            double c, double* out,
                            double eps = kBallEps) noexcept {
    if (c <= 0.0) {
        if (out != x) {
            for (std::size_t k = 0; k < d; ++k) out[k] = finite_or_zero(x[k]);
        }
        return;
    }
    const double n = norm(x, d);
    const double scale = projection_scale_from_norm(n, c, eps);
    for (std::size_t k = 0; k < d; ++k) {
        out[k] = scale * finite_or_zero(x[k]);
    }
}

struct ProjectedPairStats {
    double sx = 1.0;
    double sy = 1.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double xy = 0.0;
};

inline ProjectedPairStats projected_pair_stats(const double* x, const double* y,
                                               std::size_t d, double c) noexcept {
    ProjectedPairStats st;
    st.sx = projection_scale_from_norm(norm(x, d), c);
    st.sy = projection_scale_from_norm(norm(y, d), c);
    for (std::size_t k = 0; k < d; ++k) {
        const double xi = st.sx * finite_or_zero(x[k]);
        const double yi = st.sy * finite_or_zero(y[k]);
        st.x2 += xi * xi;
        st.y2 += yi * yi;
        st.xy += xi * yi;
    }
    return st;
}

// ---------- Poincare-ball arithmetic ----------
// Curvature is -c with c >= 0. c == 0 is the Euclidean limit.

// out = exp_0(p). Aliasing out == p is allowed.
inline void exp0(const double* p, std::size_t d, double c, double* out) noexcept {
    if (c <= 0.0) {
        if (out != p) {
            for (std::size_t k = 0; k < d; ++k) out[k] = finite_or_zero(p[k]);
        }
        return;
    }

    const double sqc = std::sqrt(c);
    const double np = norm(p, d);
    if (!std::isfinite(np)) {
        project_to_ball(p, d, c, out);
        return;
    }
    if (np < kNormEps) {
        for (std::size_t k = 0; k < d; ++k) out[k] = 0.0;
        return;
    }

    const double a = sqc * np;
    const double scale = (!std::isfinite(a) || a > 20.0)
        ? (1.0 / (sqc * np))
        : (std::tanh(a) / a);
    for (std::size_t k = 0; k < d; ++k) out[k] = scale * finite_or_zero(p[k]);
    project_to_ball(out, d, c, out);
}

// out = x (+)_c y. Inputs are projected before the algebra and the output is
// projected again, keeping the result inside the open ball even near boundary.
// Aliasing out == x or out == y is allowed.
inline void mobius_add(const double* x, const double* y, std::size_t d,
                       double c, double* out) noexcept {
    if (c <= 0.0) {
        for (std::size_t k = 0; k < d; ++k) {
            out[k] = finite_or_zero(x[k]) + finite_or_zero(y[k]);
        }
        return;
    }

    const auto st = projected_pair_stats(x, y, d, c);
    const double a = 1.0 + 2.0 * c * st.xy + c * st.y2;
    const double b = 1.0 - c * st.x2;
    double den = 1.0 + 2.0 * c * st.xy + c * c * st.x2 * st.y2;
    if (!std::isfinite(den) || std::abs(den) < kDenomEps) {
        den = (den >= 0.0 ? kDenomEps : -kDenomEps);
    }
    const double inv = 1.0 / den;

    for (std::size_t k = 0; k < d; ++k) {
        const double xi = st.sx * finite_or_zero(x[k]);
        const double yi = st.sy * finite_or_zero(y[k]);
        out[k] = (a * xi + b * yi) * inv;
    }
    project_to_ball(out, d, c, out);
}

// d_c(x, y) = 2/sqrt(c) * artanh(sqrt(c) ||(-x) (+)_c y||).
// Uses a closed form with pre-projected inputs and clips the artanh argument.
inline double poincare_distance(const double* x, const double* y, std::size_t d,
                                double c) noexcept {
    if (c <= 0.0) {
        double s = 0.0;
        for (std::size_t k = 0; k < d; ++k) {
            s = std::hypot(s, finite_or_zero(x[k]) - finite_or_zero(y[k]));
        }
        return s;
    }

    const auto st = projected_pair_stats(x, y, d, c);
    const double a = 1.0 - 2.0 * c * st.xy + c * st.y2;
    const double b = 1.0 - c * st.x2;
    double den = 1.0 - 2.0 * c * st.xy + c * c * st.x2 * st.y2;
    if (!std::isfinite(den) || std::abs(den) < kDenomEps) {
        den = (den >= 0.0 ? kDenomEps : -kDenomEps);
    }

    double nz2 = (a * a * st.x2 - 2.0 * a * b * st.xy + b * b * st.y2) /
                 (den * den);
    if (!std::isfinite(nz2) || nz2 < 0.0) nz2 = 0.0;

    const double sqc = std::sqrt(c);
    const double arg = clamp_unit_open(sqc * std::sqrt(nz2));
    if (arg <= kNormEps) return 0.0;
    return 2.0 * std::atanh(arg) / sqc;
}

inline double dist_from_origin(const double* z, std::size_t d, double c) noexcept {
    if (c <= 0.0) return norm(z, d);
    const double sqc = std::sqrt(c);
    const double n = norm(z, d);
    const double projected_n = std::isfinite(n)
        ? projection_scale_from_norm(n, c) * n
        : ball_max_norm(c);
    const double arg = clamp_unit_open(sqc * projected_n);
    if (arg <= kNormEps) return 0.0;
    return 2.0 * std::atanh(arg) / sqc;
}

// ---------- User state and online update ----------

struct UserState {
    std::vector<double> base;      // stable base b_u in T_0 M
    std::vector<double> residual;  // plastic residual r_u^(t) in T_0 M

    explicit UserState(std::size_t dim)
        : base(dim, 0.0), residual(dim, 0.0) {}

    std::size_t dim() const noexcept { return base.size(); }

    void tangent_state(double* out) const noexcept {
        const std::size_t d = base.size();
        for (std::size_t k = 0; k < d; ++k) out[k] = base[k] + residual[k];
    }

    void manifold_state(double c, double* out) const noexcept {
        tangent_state(out);
        exp0(out, base.size(), c, out);
    }
};

struct WeightedItem {
    const double* code;  // p_j in T_0 M
    double weight;
};

inline void update_user(UserState& user,
                        const double* pos_code,
                        const WeightedItem* exposed, std::size_t n_exp,
                        const WeightedItem* unknown, std::size_t n_unk,
                        double rho, double eta, double kappa) noexcept {
    const std::size_t d = user.dim();
    double* r = user.residual.data();

    for (std::size_t k = 0; k < d; ++k) {
        const double pos = pos_code ? finite_or_zero(pos_code[k]) : 0.0;
        r[k] = rho * r[k] + eta * pos;
    }

    for (std::size_t i = 0; exposed && i < n_exp; ++i) {
        if (!exposed[i].code) continue;
        const double w = eta * exposed[i].weight;
        const double* p = exposed[i].code;
        for (std::size_t k = 0; k < d; ++k) r[k] -= w * finite_or_zero(p[k]);
    }
    const double ek = eta * kappa;
    for (std::size_t i = 0; unknown && i < n_unk; ++i) {
        if (!unknown[i].code) continue;
        const double w = ek * unknown[i].weight;
        const double* p = unknown[i].code;
        for (std::size_t k = 0; k < d; ++k) r[k] -= w * finite_or_zero(p[k]);
    }
}

// s(u, i, t) = -d_c(h_u, z_i)^2 + xi * d_c(0, z_i).
inline double score(const double* h_u, const double* z_i, std::size_t d,
                    double c, double xi) noexcept {
    const double dist = poincare_distance(h_u, z_i, d, c);
    const double dist0 = dist_from_origin(z_i, d, c);
    return -dist * dist + xi * dist0;
}

struct TernaryLoss {
    double value;
    double margin_pos_exp;
    double margin_exp_unk;
};

inline TernaryLoss ternary_loss(double s_pos, double s_exp, double s_unk,
                                double delta, double gamma) noexcept {
    const double a = s_pos - s_exp - delta;
    const double b = s_exp - s_unk;
    return TernaryLoss{
        -log_sigmoid(a) - gamma * log_sigmoid(b),
        a,
        b,
    };
}

}  // namespace geometry_kernel
