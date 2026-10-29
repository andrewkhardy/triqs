// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
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
// Authors: Michel Ferrero, Laura Messio, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Implementation of the real-time/real-frequency Fourier transforms.
 *
 * @details Everything a transform needs that depends only on the pair of meshes, the number of flattened target
 * components and the direction (the FFTW plan, the phase factors before and after the FFT and the tail functions) is
 * built once and kept in a small per-thread cache. Repeated transforms on the same meshes then cost one pass over the
 * data, the FFT and a second pass, with the same arithmetic as the uncached transform.
 */

#include "./fourier.hpp"
#include "./fourier_common.hpp"

#include "../functions/functions2.hpp"

#include <array>
#include <cmath>
#include <list>
#include <mutex>
#include <string>

namespace triqs::gfs {

  namespace {
    constexpr dcomplex I{0, 1};
    inline dcomplex th_expo(double t, double a) { return (t > 0 ? -I * exp(-a * t) : (t < 0 ? 0 : -0.5 * I * exp(-a * t))); }
    inline dcomplex th_expo_neg(double t, double a) { return (t < 0 ? I * exp(a * t) : (t > 0 ? 0 : 0.5 * I * exp(a * t))); }
    inline dcomplex th_expo_inv(double w, double a) { return 1. / (w + I * a); }
    inline dcomplex th_expo_neg_inv(double w, double a) { return 1. / (w - I * a); }

    // Identifies a pair of linear meshes by their exact first value, spacing and size.
    struct mesh_pair_key {
      double in_first, in_delta, out_first, out_delta;
      long L, n_others;
      int sign;
      bool operator==(mesh_pair_key const &) const = default;
    };

    // Mesh-dependent data of one transform: an FFTW_MEASURE plan on an aligned in-place buffer of L x n_others, the
    // phase factors applied before and after the FFT, and the tail functions (built on the first call with a
    // non-vanishing tail).
    struct real_fourier_cache {
      mesh_pair_key key;
      fftw_complex *buf = nullptr;
      fftw_plan plan    = nullptr;
      array<dcomplex, 1> phase_in, phase_out;
      array<dcomplex, 1> tail_in_p, tail_in_n, tail_out_p, tail_out_n;

      explicit real_fourier_cache(mesh_pair_key const &k) : key(k), phase_in(k.L), phase_out(k.L) {
        buf    = fftw_alloc_complex(k.L * k.n_others);
        auto n = static_cast<int>(k.n_others);
        std::array<int, 1> dims{static_cast<int>(k.L)};
        {
          std::lock_guard lock{fftw_planner_mutex()};
          plan = fftw_plan_many_dft(1, dims.data(), n, buf, nullptr, n, 1, buf, nullptr, n, 1, k.sign, FFTW_MEASURE);
        }
        if (plan == nullptr) {
          fftw_free(buf);
          TRIQS_RUNTIME_ERROR << "FFTW could not create a plan for a real time/frequency Fourier transform of size " << k.L;
        }
      }
      real_fourier_cache(real_fourier_cache const &)            = delete;
      real_fourier_cache &operator=(real_fourier_cache const &) = delete;
      ~real_fourier_cache() {
        std::lock_guard lock{fftw_planner_mutex()};
        fftw_destroy_plan(plan);
        fftw_free(buf);
      }

      [[nodiscard]] dcomplex *data() { return reinterpret_cast<dcomplex *>(buf); } // NOLINT (fftw_complex is layout compatible)
    };

    // Least recently used first; a miss evicts the oldest entry beyond max_cached and calls init on the new one.
    constexpr std::size_t max_cached = 4;
    thread_local std::list<real_fourier_cache> cache_list;

    real_fourier_cache &get_cache(mesh_pair_key const &key, auto &&init) {
      for (auto it = cache_list.begin(); it != cache_list.end(); ++it) {
        if (it->key == key) {
          cache_list.splice(cache_list.begin(), cache_list, it);
          return cache_list.front();
        }
      }
      if (cache_list.size() >= max_cached) cache_list.pop_back();
      auto &c = cache_list.emplace_front(key);
      init(c);
      return c;
    }

    // Moments 1 and 2 of the tail, checked as in the uncached transform. No known_moments means a transform of the
    // raw data, i.e. a vanishing tail.
    array<dcomplex, 2> tail_moments_12(nda::array_const_view<dcomplex, 2> known_moments, long n_others, char const *direction) {
      auto mom_12 = array<dcomplex, 2>(2, n_others);
      mom_12()    = 0;
      if (known_moments.is_empty()) return mom_12;
      TRIQS_ASSERT2(known_moments.shape()[0] >= 3, std::string{" "} + direction + " Fourier transform requires known moments up to order 2.")
      double _abs_tail0 = max_element(abs(known_moments(0, range::all)));
      TRIQS_ASSERT2((_abs_tail0 < 1e-8),
                    std::string{"ERROR: "} + direction
                       + " Fourier implementation requires vanishing 0th moment\n  error is :" + std::to_string(_abs_tail0));
      mom_12 = known_moments(range(1, 3), range::all);
      return mom_12;
    }
  } // namespace

  // ------------------------ DIRECT TRANSFORM --------------------------------------------

  gf_vec_t<mesh::refreq> _fourier_impl(mesh::refreq const &w_mesh, gf_vec_cvt<mesh::retime> gt, nda::array_const_view<dcomplex, 2> known_moments) {

    long n_others = second_dim(gt.data());
    auto mom_12   = tail_moments_12(known_moments, n_others, "Direct RealTime");

    long L = gt.mesh().size();
    if (w_mesh.size() != L) TRIQS_RUNTIME_ERROR << "Meshes are different";
    double test = std::abs(gt.mesh().delta() * w_mesh.delta() * static_cast<double>(L) / (2 * M_PI) - 1);
    if (test > 1.e-10) TRIQS_RUNTIME_ERROR << "Meshes are not compatible";

    auto const &t_mesh = gt.mesh();
    const double tmin  = t_mesh.t_min();
    const double wmin  = w_mesh.w_min();
    //a is a number very larger than delta_w and very smaller than wmax-wmin, used in the tail computation
    const double a = w_mesh.delta() * std::sqrt(static_cast<double>(L));

    auto key = mesh_pair_key{t_mesh.to_value(0), t_mesh.delta(), w_mesh.to_value(0), w_mesh.delta(), L, n_others, FFTW_BACKWARD};
    auto &c  = get_cache(key, [&](real_fourier_cache &entry) {
      for (long i = 0; i < L; ++i) {
        double t           = t_mesh.to_value(i);
        double w           = w_mesh.to_value(i);
        entry.phase_in(i)  = std::exp(I * t * wmin);
        entry.phase_out(i) = t_mesh.delta() * std::exp(I * (w - wmin) * tmin);
      }
    });

    auto _        = range::all;
    auto m1       = mom_12(0, _);
    auto m2       = mom_12(1, _);
    bool has_tail = max_element(abs(mom_12)) > 0;

    array<dcomplex, 1> a1 = (m1 + I * m2 / a) / 2., a2 = (m1 - I * m2 / a) / 2.;

    if (has_tail and c.tail_in_p.size() != L) {
      c.tail_in_p.resize(L), c.tail_in_n.resize(L), c.tail_out_p.resize(L), c.tail_out_n.resize(L);
      for (long i = 0; i < L; ++i) {
        c.tail_in_p(i)  = th_expo(t_mesh.to_value(i), a);
        c.tail_in_n(i)  = th_expo_neg(t_mesh.to_value(i), a);
        c.tail_out_p(i) = th_expo_inv(w_mesh.to_value(i), a);
        c.tail_out_n(i) = th_expo_neg_inv(w_mesh.to_value(i), a);
      }
    }

    auto const &gin = gt.data();
    auto *buf       = c.data();
    for (long i = 0; i < L; ++i)
      for (long k = 0; k < n_others; ++k) {
        auto g                  = has_tail ? gin(i, k) - (a1(k) * c.tail_in_p(i) + a2(k) * c.tail_in_n(i)) : gin(i, k);
        buf[(i * n_others) + k] = g * c.phase_in(i);
      }

    fftw_execute(c.plan);

    auto gw    = gf_vec_t<mesh::refreq>{w_mesh, {n_others}};
    auto &gout = gw.data();
    for (long i = 0; i < L; ++i)
      for (long k = 0; k < n_others; ++k) {
        gout(i, k) = c.phase_out(i) * buf[(i * n_others) + k];
        if (has_tail) gout(i, k) = gout(i, k) + a1(k) * c.tail_out_p(i) + a2(k) * c.tail_out_n(i);
      }

    return gw;
  }

  // ------------------------ INVERSE TRANSFORM --------------------------------------------

  gf_vec_t<mesh::retime> _fourier_impl(mesh::retime const &t_mesh, gf_vec_cvt<mesh::refreq> gw, nda::array_const_view<dcomplex, 2> known_moments) {

    long n_others = second_dim(gw.data());
    auto mom_12   = tail_moments_12(known_moments, n_others, "Inverse Real-Frequency");

    long L = gw.mesh().size();
    if (L != t_mesh.size()) TRIQS_RUNTIME_ERROR << "Meshes are of different size: " << L << " vs " << t_mesh.size();
    double test = std::abs(t_mesh.delta() * gw.mesh().delta() * static_cast<double>(L) / (2 * M_PI) - 1);
    if (test > 1.e-10) TRIQS_RUNTIME_ERROR << "Meshes are not compatible";

    auto const &w_mesh = gw.mesh();
    const double tmin  = t_mesh.t_min();
    const double wmin  = w_mesh.w_min();
    //a is a number very larger than delta_w and very smaller than wmax-wmin, used in the tail computation
    const double a    = w_mesh.delta() * std::sqrt(static_cast<double>(L));
    const double corr = 1.0 / (t_mesh.delta() * static_cast<double>(L));

    auto key = mesh_pair_key{w_mesh.to_value(0), w_mesh.delta(), t_mesh.to_value(0), t_mesh.delta(), L, n_others, FFTW_FORWARD};
    auto &c  = get_cache(key, [&](real_fourier_cache &entry) {
      for (long i = 0; i < L; ++i) {
        double w           = w_mesh.to_value(i);
        double t           = t_mesh.to_value(i);
        entry.phase_in(i)  = std::exp(-I * w * tmin);
        entry.phase_out(i) = corr * std::exp(I * wmin * (tmin - t));
      }
    });

    auto _        = range::all;
    auto m1       = mom_12(0, _);
    auto m2       = mom_12(1, _);
    bool has_tail = max_element(abs(mom_12)) > 0;

    array<dcomplex, 1> a1 = (m1 + I * m2 / a) / 2., a2 = (m1 - I * m2 / a) / 2.;

    if (has_tail and c.tail_in_p.size() != L) {
      c.tail_in_p.resize(L), c.tail_in_n.resize(L), c.tail_out_p.resize(L), c.tail_out_n.resize(L);
      for (long i = 0; i < L; ++i) {
        c.tail_in_p(i)  = th_expo_inv(w_mesh.to_value(i), a);
        c.tail_in_n(i)  = th_expo_neg_inv(w_mesh.to_value(i), a);
        c.tail_out_p(i) = th_expo(t_mesh.to_value(i), a);
        c.tail_out_n(i) = th_expo_neg(t_mesh.to_value(i), a);
      }
    }

    auto const &gin = gw.data();
    auto *buf       = c.data();
    for (long i = 0; i < L; ++i)
      for (long k = 0; k < n_others; ++k) {
        auto g                  = has_tail ? gin(i, k) - a1(k) * c.tail_in_p(i) - a2(k) * c.tail_in_n(i) : gin(i, k);
        buf[(i * n_others) + k] = g * c.phase_in(i);
      }

    fftw_execute(c.plan);

    auto gt    = gf_vec_t<mesh::retime>{t_mesh, {n_others}};
    auto &gout = gt.data();
    for (long i = 0; i < L; ++i)
      for (long k = 0; k < n_others; ++k) {
        gout(i, k) = c.phase_out(i) * buf[(i * n_others) + k];
        if (has_tail) gout(i, k) = gout(i, k) + a1(k) * c.tail_out_p(i) + a2(k) * c.tail_out_n(i);
      }

    return gt;
  }

} // namespace triqs::gfs
