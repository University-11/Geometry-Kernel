// Build:
//   g++ -O3 -std=c++17 -Wall -Wextra -I Geometry-Kernel/include Geometry-Kernel/tests/geomemory_test.cpp -o geomemory_test
//
// Verifies the laptop-first GeoMemory kernel: Poincare arithmetic, the
// stable-base + plastic-residual UserState, the online tangent-space update,
// and the ternary loss that splits exposure refusal from epistemic ignorance.

#include "geometry_kernel/geomemory.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gm = geometry_kernel;

namespace {

int g_failed = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failed;
    }
}

void check_close(double got, double want, double tol, const char* msg) {
    if (std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s  got=%.12g want=%.12g (|d|=%.3g, tol=%.3g)\n",
                     msg, got, want, std::abs(got - want), tol);
        ++g_failed;
    }
}

void test_euclidean_limit() {
    double x[2] = {1.0, 2.0};
    double y[2] = {3.0, 4.0};
    double out[2];
    gm::mobius_add(x, y, 2, /*c=*/0.0, out);
    check_close(out[0], 4.0, 1e-15, "mobius_add c=0 [0]");
    check_close(out[1], 6.0, 1e-15, "mobius_add c=0 [1]");
    gm::exp0(x, 2, 0.0, out);
    check_close(out[0], 1.0, 1e-15, "exp0 c=0 [0]");
    check_close(out[1], 2.0, 1e-15, "exp0 c=0 [1]");
    double zero[2] = {0.0, 0.0};
    double pt[2] = {3.0, 4.0};
    check_close(gm::poincare_distance(zero, pt, 2, 0.0), 5.0, 1e-12,
                "poincare_distance c=0");
}

void test_distance_symmetry_and_zero() {
    double x[3] = {0.10, -0.20, 0.05};
    double y[3] = {0.20,  0.05, -0.10};
    const double dxy = gm::poincare_distance(x, y, 3, 1.0);
    const double dyx = gm::poincare_distance(y, x, 3, 1.0);
    check_close(dxy, dyx, 1e-12, "poincare_distance symmetry");
    check_close(gm::poincare_distance(x, x, 3, 1.0), 0.0, 1e-12,
                "poincare_distance(x, x) = 0");
}

void test_radial_isometry() {
    // In the conformal Poincare-ball convention used by the PDF (and matching
    // tests/test_poincare.py), exp_0 is a *radial* isometry up to the conformal
    // factor at the origin: d_c(0, exp_0(p)) = 2 * ||p||.
    double p[2] = {0.3, 0.4};   // ||p|| = 0.5
    double z[2];
    gm::exp0(p, 2, 1.0, z);
    double zero[2] = {0.0, 0.0};
    check_close(gm::poincare_distance(zero, z, 2, 1.0), 1.0, 1e-12,
                "radial isometry: d_c(0, exp_0(p)) == 2 ||p||");
    check_close(gm::dist_from_origin(z, 2, 1.0), 1.0, 1e-12,
                "dist_from_origin closed form");
    // And ||z|| matches the exp_0 radial closed form: tanh(sqrt(c) ||p||)/sqrt(c).
    check_close(gm::norm(z, 2), std::tanh(0.5), 1e-12,
                "exp_0 radial magnitude");
}

void test_boundary_projection_and_clipping() {
    const double c = 1.0;
    double outside[2] = {10.0, 0.0};
    double projected[2];
    gm::project_to_ball(outside, 2, c, projected);
    check(gm::norm(projected, 2) < 1.0, "projection keeps point inside unit ball");

    double near_a[2] = {1.0 - 1e-16, 0.0};
    double near_b[2] = {0.0, 1.0 - 1e-16};
    const double d = gm::poincare_distance(near_a, near_b, 2, c);
    check(std::isfinite(d), "distance near boundary stays finite");

    double sum[2];
    gm::mobius_add(outside, near_b, 2, c, sum);
    check(gm::norm(sum, 2) < 1.0, "mobius_add output is clipped into open ball");
}

void test_exp0_large_tangent_is_finite_and_inside_ball() {
    double p[2] = {1e150, -1e150};
    double z[2];
    gm::exp0(p, 2, 1.0, z);
    check(std::isfinite(z[0]) && std::isfinite(z[1]), "exp0 large tangent finite");
    check(gm::norm(z, 2) < 1.0, "exp0 large tangent clipped inside ball");
}

void test_user_state_default() {
    gm::UserState u(8);
    check(u.dim() == 8, "UserState dim");
    double tang[8];
    u.tangent_state(tang);
    for (int k = 0; k < 8; ++k) check(tang[k] == 0.0, "tangent state starts at 0");
    double mani[8];
    u.manifold_state(1.0, mani);
    for (int k = 0; k < 8; ++k) check(mani[k] == 0.0, "manifold state starts at 0");
}

void test_update_user_pos_only() {
    // With no negatives, r <- rho * r + eta * p_pos.
    gm::UserState u(4);
    double pos[4] = {0.10, -0.20, 0.30, 0.05};
    gm::update_user(u, pos, nullptr, 0, nullptr, 0,
                    /*rho=*/0.9, /*eta=*/0.5, /*kappa=*/0.5);
    for (int k = 0; k < 4; ++k) {
        check_close(u.residual[k], 0.5 * pos[k], 1e-15, "first step = eta * pos");
    }
    // Second step: r <- 0.9 * r_prev + 0.5 * pos.
    std::vector<double> prev = u.residual;
    gm::update_user(u, pos, nullptr, 0, nullptr, 0, 0.9, 0.5, 0.5);
    for (int k = 0; k < 4; ++k) {
        check_close(u.residual[k], 0.9 * prev[k] + 0.5 * pos[k], 1e-15,
                    "second step decays prev and adds eta * pos");
    }
}

void test_update_user_exposed_vs_unknown() {
    // kappa downweights unknown negatives. Same vector contributed once as
    // exposed (weight 1) and once as unknown (weight 1) should net to (1 - kappa).
    gm::UserState u(3);
    double pos[3] = {0.0, 0.0, 0.0};         // no positive signal
    double q[3]   = {1.0, 0.0, 0.0};         // a single direction
    gm::WeightedItem exp_neg{q, 1.0};
    gm::WeightedItem unk_neg{q, 1.0};
    gm::update_user(u, pos, &exp_neg, 1, &unk_neg, 1,
                    /*rho=*/0.0, /*eta=*/1.0, /*kappa=*/0.25);
    // r = 0 + 1*(0) - 1*1*q - 1*0.25*1*q = -1.25 * q.
    check_close(u.residual[0], -1.25, 1e-15, "exposed + 0.25*unknown directions");
    check_close(u.residual[1],  0.0,  1e-15, "no leak in y");
    check_close(u.residual[2],  0.0,  1e-15, "no leak in z");
}

void test_update_user_drift_toward_positive() {
    // After repeated reinforcement of the same chosen item with weak negative
    // pressure, the user score on that item must rise.
    const std::size_t d = 16;
    gm::UserState u(d);

    std::vector<double> pos(d, 0.0), neg(d, 0.0);
    pos[0] = 0.30; pos[3] = -0.15; pos[7] = 0.10;
    neg[1] = 0.25; neg[5] =  0.10;

    gm::WeightedItem exp_neg{neg.data(), 1.0};

    std::vector<double> z_pos(d), z_neg(d), h(d);
    gm::exp0(pos.data(), d, 1.0, z_pos.data());
    gm::exp0(neg.data(), d, 1.0, z_neg.data());

    u.manifold_state(1.0, h.data());
    const double s_pos_0 = gm::score(h.data(), z_pos.data(), d, 1.0, 0.0);
    const double s_neg_0 = gm::score(h.data(), z_neg.data(), d, 1.0, 0.0);

    for (int t = 0; t < 50; ++t) {
        gm::update_user(u, pos.data(), &exp_neg, 1, nullptr, 0,
                        /*rho=*/0.95, /*eta=*/0.05, /*kappa=*/0.5);
    }
    u.manifold_state(1.0, h.data());
    const double s_pos_1 = gm::score(h.data(), z_pos.data(), d, 1.0, 0.0);
    const double s_neg_1 = gm::score(h.data(), z_neg.data(), d, 1.0, 0.0);

    check(s_pos_1 > s_pos_0, "positive score increases after reinforcement");
    check(s_neg_1 < s_neg_0, "exposed-negative score decreases");
    check(s_pos_1 - s_neg_1 > s_pos_0 - s_neg_0,
          "ranking margin pos vs exposed-neg widens");
}

void test_ternary_loss_basic() {
    // gamma = 0 => the loss collapses to a margin BPR on (pos, exposed).
    const auto a = gm::ternary_loss(/*s_pos=*/2.0, /*s_exp=*/0.0,
                                    /*s_unk=*/0.0, /*delta=*/0.5, /*gamma=*/0.0);
    const double want_a = -gm::log_sigmoid(2.0 - 0.0 - 0.5);
    check_close(a.value, want_a, 1e-15, "ternary collapses to BPR at gamma=0");
    check_close(a.margin_pos_exp, 1.5, 1e-15, "diagnostic margin pos-exp");
    check_close(a.margin_exp_unk, 0.0, 1e-15, "diagnostic margin exp-unk");

    // Loss must decrease as the chosen item pulls ahead.
    const auto bad  = gm::ternary_loss(0.0, 0.0, 0.0, 0.5, 1.0);
    const auto good = gm::ternary_loss(3.0, 0.0, 1.0, 0.5, 1.0);
    check(good.value < bad.value, "loss drops when scores improve");
    check(good.value > 0.0, "loss is non-negative");
}

void test_ternary_loss_separation() {
    // The whole point of the ternary form: changing s_unk while holding s_pos,
    // s_exp fixed must move the loss only through the second term.
    const double s_pos = 1.5, s_exp = 0.2, delta = 0.3, gamma = 0.7;
    const auto a = gm::ternary_loss(s_pos, s_exp, /*s_unk=*/-1.0, delta, gamma);
    const auto b = gm::ternary_loss(s_pos, s_exp, /*s_unk=*/+0.5, delta, gamma);
    const double diff = a.value - b.value;
    const double want = gamma * (gm::log_sigmoid(s_exp - 0.5) -
                                 gm::log_sigmoid(s_exp + 1.0));
    check_close(diff, want, 1e-12,
                "delta(loss) on s_unk equals gamma * delta(log sigmoid)");
}

void bench_update_user() {
    // Order-of-magnitude sanity check that the hot loop is cache-friendly.
    const std::size_t d = 32;
    const std::size_t n_items = 4096;
    std::vector<double> codes(n_items * d);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        codes[i] = std::sin(0.001 * static_cast<double>(i));
    }
    std::vector<gm::WeightedItem> exp_negs(8);
    std::vector<gm::WeightedItem> unk_negs(16);
    for (std::size_t i = 0; i < exp_negs.size(); ++i)
        exp_negs[i] = {codes.data() + (i * 17 % n_items) * d, 1.0 / exp_negs.size()};
    for (std::size_t i = 0; i < unk_negs.size(); ++i)
        unk_negs[i] = {codes.data() + (i * 31 % n_items) * d, 1.0 / unk_negs.size()};

    gm::UserState u(d);
    const int iters = 200000;
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < iters; ++t) {
        const double* pos = codes.data() + (t * 7 % n_items) * d;
        gm::update_user(u, pos, exp_negs.data(), exp_negs.size(),
                        unk_negs.data(), unk_negs.size(),
                        0.97, 0.02, 0.5);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ns_per_step =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
    std::printf("update_user: d=%zu, |E-|=%zu, |U-|=%zu  ->  %.0f ns/step\n",
                d, exp_negs.size(), unk_negs.size(), ns_per_step);
}

}  // namespace

int main() {
    test_euclidean_limit();
    test_distance_symmetry_and_zero();
    test_radial_isometry();
    test_boundary_projection_and_clipping();
    test_exp0_large_tangent_is_finite_and_inside_ball();
    test_user_state_default();
    test_update_user_pos_only();
    test_update_user_exposed_vs_unknown();
    test_update_user_drift_toward_positive();
    test_ternary_loss_basic();
    test_ternary_loss_separation();

    if (g_failed != 0) {
        std::fprintf(stderr, "%d test(s) failed.\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("All %s tests passed.\n", "geomemory");
    bench_update_user();
    return EXIT_SUCCESS;
}
