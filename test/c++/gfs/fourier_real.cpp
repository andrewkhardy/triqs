// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2023 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt
//
// Authors: Olivier Parcollet, Nils Wentzell

#include <triqs/test_tools/gfs.hpp>

double lorentzian(double w, double E) { return 2 * E / (w * w + E * E); };
std::complex<double> lorentzian_inverse(double t, double E) { return std::exp(-E * std::abs(t)); };
double theta(double x) { return x > 0 ? 1.0 : (x < 0 ? 0.0 : 0.5); };

// Generic Fourier test function for different ranks
template <int TARGET_RANK> void test_fourier() {
  double precision = 1e-4;
  double w_max     = 40;
  int Nw           = 1001;
  double E         = 1;

  std::array<long, TARGET_RANK> shape{};

  if constexpr (TARGET_RANK == 2) // Matrix_valued
    shape = make_shape(2, 2);
  else if constexpr (TARGET_RANK == 3) // Tensor_valued<3>
    shape = make_shape(2, 2, 2);
  else if constexpr (TARGET_RANK == 4) // Tensor_valued<4>
    shape = make_shape(2, 2, 2, 2);

  using target_t = typename _target_from_type_rank<dcomplex, TARGET_RANK>::type;

  auto w_mesh = mesh::refreq{-w_max, w_max, Nw};
  auto t_mesh = make_adjoint_mesh(w_mesh);
  auto Gw1    = gf<refreq, target_t>{w_mesh, shape};

  // === Fourier of a simple Lorentzian ===

  for (auto w : w_mesh) Gw1[w] = lorentzian(w, E);

  auto [tail, err] = fit_tail(Gw1);
  auto Gt1         = make_gf_from_fourier(Gw1, t_mesh, tail);

  // verification that TF(TF^-1)=Id
  auto Gw1b = make_gf_from_fourier(Gt1, w_mesh, tail);
  EXPECT_GF_NEAR(Gw1, Gw1b, precision);

  // Comparision against exact result
  auto Gt1_exact = Gt1;
  for (auto t : t_mesh) Gt1_exact[t] = lorentzian_inverse(t, E);
  EXPECT_GF_NEAR(Gt1, Gt1_exact, precision);

  // === Fourier of a Product of theta function and exponential ===

  for (auto t : t_mesh) Gt1[t] = 0.5i * (lorentzian_inverse(-t, E) * theta(-t) - lorentzian_inverse(t, E) * theta(t));

  auto known_moments = make_zero_tail(Gt1);

  if constexpr (TARGET_RANK == 2) { // Matrix valued case needs special treatment
    known_moments(1, 0, 0) = 1.;
    known_moments(1, 1, 1) = 1.;
  } else
    known_moments(1, ellipsis()) = 1.;

  Gw1() = fourier(Gt1, known_moments);

  // verification that TF(TF^-1)=Id
  auto Gt1b = make_gf_from_fourier(Gw1, t_mesh, known_moments);
  EXPECT_GF_NEAR(Gt1, Gt1b, precision);

  // Comparision against exact result
  auto Gw1_exact = Gw1;
  for (auto w : w_mesh) Gw1_exact[w] = 0.5 / (w + E * 1i) + 0.5 / (w - E * 1i);
  EXPECT_GF_NEAR(Gw1, Gw1_exact, precision);
}

TEST(FourierReal, Scalar) { test_fourier<0>(); }
TEST(FourierReal, Matrix) { test_fourier<2>(); }
TEST(FourierReal, Tensor3) { test_fourier<3>(); }
TEST(FourierReal, Tensor4) { test_fourier<4>(); }

// The plans and phase factors are cached per mesh pair: check the transforms without tail against direct sums, that a
// repeated call reproduces the first one exactly, and that the result is unchanged once the cache has cycled through
// more mesh pairs than it holds.
TEST(FourierReal, CachedAgainstDirectSum) {
  auto w_mesh = mesh::refreq{-10, 10, 401};
  auto t_mesh = make_adjoint_mesh(w_mesh);
  auto L      = w_mesh.size();

  auto Gw = gf<refreq, matrix_valued>{w_mesh, {2, 2}};
  for (long n = 0; n < L; ++n)
    for (long a : range(2))
      for (long b : range(2)) Gw.data()(n, a, b) = 1.0 / (w_mesh.to_value(n) - 0.5 * double(a - b) + 1i * (0.2 + 0.1 * double(a + b)));

  auto Gt  = make_gf_from_fourier(Gw, t_mesh);
  auto Gw2 = make_gf_from_fourier(Gt, w_mesh);

  // g(t_j) = 1/(dt L) sum_n g(w_n) e^{-i w_n t_j} and g(w_j) = dt sum_n g(t_n) e^{i w_j t_n}
  double corr = 1.0 / (t_mesh.delta() * static_cast<double>(L));
  for (long j : {0L, 1L, 77L, 200L, 399L, 400L})
    for (long a : range(2))
      for (long b : range(2)) {
        dcomplex st = 0, sw = 0;
        for (long n = 0; n < L; ++n) {
          st += Gw.data()(n, a, b) * std::exp(-1i * w_mesh.to_value(n) * t_mesh.to_value(j));
          sw += Gt.data()(n, a, b) * std::exp(1i * w_mesh.to_value(j) * t_mesh.to_value(n));
        }
        EXPECT_NEAR(std::abs(corr * st - Gt.data()(j, a, b)), 0, 1e-12);
        EXPECT_NEAR(std::abs(t_mesh.delta() * sw - Gw2.data()(j, a, b)), 0, 1e-12);
      }

  // Cache hit: same plan and phase factors, identical result
  auto Gt_hit = make_gf_from_fourier(Gw, t_mesh);
  EXPECT_EQ(max_element(abs(Gt_hit.data() - Gt.data())), 0.0);

  // Cycle the cache through more mesh pairs than it holds, then transform on the first meshes again
  for (long N : {101, 103, 105, 107, 109}) {
    auto w_other   = mesh::refreq{-10, 10, N};
    auto g_other   = gf<refreq, matrix_valued>{w_other, {2, 2}};
    g_other.data() = 0;
    auto g_other_t = make_gf_from_fourier(g_other, make_adjoint_mesh(w_other));
    auto g_other_w = make_gf_from_fourier(g_other_t, w_other);
    EXPECT_EQ(max_element(abs(g_other_w.data())), 0.0);
  }
  auto Gt_again = make_gf_from_fourier(Gw, t_mesh);
  EXPECT_ARRAY_NEAR(Gt_again.data(), Gt.data(), 1e-13);
}

MAKE_MAIN;
