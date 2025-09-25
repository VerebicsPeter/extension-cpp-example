#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

using namespace torch::indexing;

#include <tuple>
#include <vector>

extern "C"
{
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject *PyInit__C(void)
  {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
  }
}

namespace torchpoly_cpp
{

  /* NOTE: highest degree first (not numpy equvivalent: lowest degree first) */
  at::Tensor poly_mul(const at::Tensor &p1, const at::Tensor &p2)
  {
    TORCH_CHECK(p1.device() == p2.device(), "All tensors must have same device");
    TORCH_CHECK(p1.device().is_cpu());
    TORCH_CHECK(p1.dtype() == p2.dtype(), "All tensors must have same dtype");
    // Proceed with the scalar type
    return AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "poly_mul",
    [&]() {
      at::Tensor p1_contig = p1.contiguous();
      at::Tensor p2_contig = p2.contiguous();
      // n+1 + m+1 - 1 = n+m+1
      at::Tensor result = torch::zeros({p1_contig.size(0) + p2_contig.size(0) - 1}, p1_contig.options());

      const scalar_t *p1_ptr = p1_contig.data_ptr<scalar_t>();
      const scalar_t *p2_ptr = p2_contig.data_ptr<scalar_t>();
      scalar_t *result_ptr = result.data_ptr<scalar_t>();

      for (int64_t i = 0; i < p1_contig.numel(); i++)
      {
        for (int64_t j = 0; j < p2_contig.numel(); j++)
        {
          result_ptr[i + j] += p1_ptr[i] * p2_ptr[j];
        }
      }
      return result;
    });
  }

  /* NOTE: highest degree first (not numpy equvivalent: lowest degree first) */
  at::Tensor poly_fromroots(const at::Tensor &roots)
  {
    TORCH_CHECK(roots.device().is_cpu());

    return AT_DISPATCH_FLOATING_TYPES(roots.scalar_type(), "poly_fromroots",
    [&]() {
      at::Tensor roots_contig = roots.contiguous();
      const scalar_t *roots_ptr = roots_contig.data_ptr<scalar_t>();

      at::Tensor coeffs;
      at::Tensor Q;
      for (int64_t k = 0; k < roots_contig.numel(); k++)
      {
        Q = torch::tensor({scalar_t(1.0), -roots_ptr[k]}, roots_contig.options());
        if (k == 0)
        {
          coeffs = Q;
        }
        else
        {
          coeffs = poly_mul(Q, coeffs);
        }
      }
      return coeffs;
    });
  }

  /* NOTE: highest degree first (not numpy equvivalent: lowest degree first) */
  at::Tensor poly_val(const at::Tensor &coeffs, const at::Tensor &x)
  {
    TORCH_CHECK(coeffs.device() == x.device(), "All tensors must have same device");
    TORCH_CHECK(coeffs.device().is_cpu());
    TORCH_CHECK(coeffs.dtype() == x.dtype(), "All tensors must have same dtype");

    return AT_DISPATCH_FLOATING_TYPES(coeffs.scalar_type(), "poly_val",
    [&]() {
      at::Tensor coeffs_contig = coeffs.contiguous();
      const scalar_t *coeffs_ptr = coeffs_contig.data_ptr<scalar_t>();

      at::Tensor result = torch::zeros_like(x, x.options());
      
      for (int64_t i = 0; i < coeffs_contig.numel(); i++)
      {
        result = result * x + coeffs_ptr[i];
      }
      return result;
    });
  }

  /* Compute the derivative of a polynomial given its coefficients */
  at::Tensor poly_der(const at::Tensor &coeffs)
  {
    TORCH_CHECK(coeffs.device().is_cpu());

    const int64_t N  = coeffs.size(0) - 1;
    at::Tensor order = torch::arange(N, 0, -1, coeffs.options());
    at::Tensor slice = coeffs.slice(0, 0, N);
    return slice * order;
  }

  // Defines the operators
  TORCH_LIBRARY(torchpoly_cpp, m)
  {
    m.def("poly_mul(Tensor p1, Tensor p2) -> Tensor");
    m.def("poly_fromroots(Tensor roots) -> Tensor");
    m.def("poly_val(Tensor coeffs, Tensor x) -> Tensor");
    m.def("poly_der(Tensor coeffs) -> Tensor");
  }

  // Registers CPU implementations
  TORCH_LIBRARY_IMPL(torchpoly_cpp, CPU, m)
  {
    m.impl("poly_mul", &poly_mul);
    m.impl("poly_fromroots", &poly_fromroots);
    m.impl("poly_val", &poly_val);
    m.impl("poly_der", &poly_der);
  }

}

namespace torchwavelets_cpp
{

  /*
  beta : imag part of one of the poles of R
  bmin : minimum absolute value of imaginary part of R's poles
  */
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  _Q(const at::Tensor &x, const at::Scalar &a, const at::Scalar &beta, const at::Scalar &bmin)
  {
    auto device = x.device();
    TORCH_CHECK(device.is_cpu());
    auto a_ = at::scalar_to_tensor(a, device).to(x.dtype());
    auto beta_ = at::scalar_to_tensor(beta, device).to(x.dtype());
    auto bmin_ = at::scalar_to_tensor(bmin, device).to(x.dtype());

    auto b_ = beta_ * beta_ + bmin_; auto db = 2*beta_; // b_ and derivative of b_ for beta
    auto b_sqr = b_*b_;
    auto a_sqr = a_*a_;

    auto twice_b_sqr_minus_a_sqr = (2 * (b_sqr - a_sqr));
    at::Tensor Qf = at::pow(x, 4) + at::pow(x, 2) * twice_b_sqr_minus_a_sqr + (at::pow(a_, 4)) + (2 * a_sqr * b_sqr) + (at::pow(b_, 4));
    at::Tensor dQx = 4 * at::pow(x, 3) + 2 * x * (  twice_b_sqr_minus_a_sqr  );
    at::Tensor dQa = 4 *       (-at::pow(x, 2) * a_ + at::pow(a_, 3) + a_ * b_sqr);
    at::Tensor dQb = (4 * db) * (at::pow(x, 2) * b_ + at::pow(b_, 3) + b_ * a_sqr);

    return {Qf, dQx, dQa, dQb};
  }

  /*
  x : dilated, translated datapoints of the effective support of the function
  a : real part one of the poles of R
  beta : imag part one of the poles of R
  bmin : minimum absolute value of imaginary part of R's poles
  */
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  _R(const at::Tensor &x, const at::Scalar &a, const at::Scalar &beta, const at::Scalar &bmin)
  {
    TORCH_CHECK(x.device().is_cpu());

    auto [Qf, dQx, dQa, dQb] = _Q(x, a, beta, bmin);
    at::Tensor Rf = at::pow(Qf, -1);
    at::Tensor inv_sqr_Qf = -1 * at::pow(Qf, -2);
    at::Tensor dRx = inv_sqr_Qf * dQx;
    at::Tensor dRa = inv_sqr_Qf * dQa;
    at::Tensor dRb = inv_sqr_Qf * dQb;
    return {Rf, dRx, dRa, dRb};
  }
  
  /*
  x : dilated, translated datapoints of the effective support of the function
  ak : real part of the poles of R
  betak : imag part of the poles of R
  pk : zeros of the polynom on the positive half-space
  bmin : minimum absolute value of imaginary part of R's poles
  sigma : parameter of the Gaussian function
  */
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  psi_fun(
      const at::Tensor &x,
      const at::Tensor &ak,
      const at::Tensor &betak,
      const at::Tensor &pk,
      const at::Scalar &bmin,
      const at::Scalar &sigma)
  {
    TORCH_CHECK(x.device().is_cpu());
    TORCH_CHECK(ak.device().is_cpu());
    TORCH_CHECK(pk.device().is_cpu());
    TORCH_CHECK(betak.device().is_cpu());

    const int64_t N = ak.numel();
    const int64_t M = pk.numel();
    const int64_t L = x.numel();

    at::Tensor Rfun = torch::ones({L}, x.options());
    at::Tensor r_k = torch::zeros({N, L}, x.options());
    at::Tensor dRx_k = torch::zeros({N, L}, x.options());
    at::Tensor dRa_k = torch::zeros({N, L}, x.options());
    at::Tensor dRb_k = torch::zeros({N, L}, x.options());

    at::Tensor zeros = torch::cat({pk, -pk, torch::tensor({0.0}, pk.options())});

    at::Tensor Palg = torchpoly_cpp::poly_fromroots(zeros);
    at::Tensor dPalg = torchpoly_cpp::poly_der(Palg);
    at::Tensor Pf = torchpoly_cpp::poly_val(Palg, x);
    at::Tensor dPx = torchpoly_cpp::poly_val(dPalg, x);

    /* Construct the rational term R */
    for (int64_t k = 0; k < N; k++)
    {
      auto [r, rx, ra, rb] = _R(x, ak[k].item(), betak[k].item(), bmin);
      Rfun *= r;
      r_k[k] = r;
      dRx_k[k] = rx;
      dRa_k[k] = ra;
      dRb_k[k] = rb;
    }

    /* Construct R's derivative w.r.t. x */
    at::Tensor dRx = torch::zeros_like(x, x.options());
    for (int64_t k = 0; k < N; k++)
    {
      auto rr = torch::cat({r_k.slice(0,0,k), r_k.slice(0,k+1)}); // erase kth row because of chain rule of derivatives
      dRx += dRx_k[k] * torch::prod(rr, 0);
    }

    auto sigma_val = at::scalar_to_tensor(sigma).to(x.dtype());
    auto sigma_sqr = sigma_val*sigma_val;
    auto exp_neg_x_over_sigma_sqr = torch::exp(-at::pow(x,2)/sigma_sqr);

    /*  Construct the mother wavelet and derivatives */
    at::Tensor Psi = Pf*Rfun*exp_neg_x_over_sigma_sqr;

    // Derivatives w.r.t. x
    at::Tensor dPsix = dPx*Rfun*exp_neg_x_over_sigma_sqr + Pf*dRx*exp_neg_x_over_sigma_sqr - 2*x/sigma_sqr*Psi;
    // Derivatives w.r.t. a,b
    at::Tensor dPsia = torch::zeros({N, L}, x.options());
    at::Tensor dPsib = torch::zeros({N, L}, x.options());

    for (int64_t k = 0; k < N; k++)
    {
      auto rr = torch::cat({r_k.slice(0,0,k), r_k.slice(0,k+1)}); // erase kth row because of partial derivatives
      auto tmp = torch::prod(rr, 0)*Pf*exp_neg_x_over_sigma_sqr;
      dPsia[k] = dRa_k[k]*tmp;
      dPsib[k] = dRb_k[k]*tmp;
    }

    // Derivatives w.r.t. p
    at::Tensor dPsip = torch::zeros({M, L}, x.options());
    for (int64_t k = 0; k < M; k++)
    {
      at::Tensor roots, dPp;
      dPp = -( Pf/( (x - pk[k])*(x + pk[k]) ) )*2*pk[k];
      roots = torch::cat({pk.slice(0,0,k), pk.slice(0,k+1)});
      roots = torch::cat({roots, -roots, torch::tensor({0.0f}, roots.options())});
      auto Pcurr = torchpoly_cpp::poly_fromroots(roots);
      Pf = torchpoly_cpp::poly_val(Pcurr, x);
      dPp = -Pf*2*pk[k];
      dPsip[k] = dPp*Rfun*exp_neg_x_over_sigma_sqr;
    }

    // Derivatives w.r.t. sigma
    auto dPsiSigma = Psi*2*at::pow(x,2)*at::pow(sigma_val, -3);

    return {Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma};
  }

  /*
  p : number of zeros in P
  r : number of poles in R
  n : number of wavelet coefficients
  bmin : minimum absolute value of imaginary part of R's poles
  alpha : (p1, ..., pp, r0real, r0imag, ..., rrreal, rrimag, s1, x1, s2, x2, ..., sn, xn, sigma)
  */
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  adaRatGaussWav(
      c10::SymInt n,
      const at::Tensor& t,
      const at::Tensor& params,
      c10::SymInt p,
      c10::SymInt r,
      const at::Scalar& bmin,
      const at::Scalar& smin=0.01,
      bool s_square=false,
      at::ScalarType dtype=at::kFloat,
      at::Device device=at::kCPU
  );

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  adaRatGaussWav(
      c10::SymInt n,
      const at::Tensor& t,
      const at::Tensor& params,
      c10::SymInt p,
      c10::SymInt r,
      const at::Scalar& bmin,
      const at::Scalar& smin,
      bool s_square,
      at::ScalarType dtype,
      at::Device device)
  {
    TORCH_CHECK(t.dtype()  == dtype  && params.dtype()  == dtype );
    TORCH_CHECK(t.device() == device && params.device() == device);

    at::Tensor alpha = params;

    // Some useful constants for indexing
    auto polebeg = p.expect_int(); // p+1 in matlab, because start index is 1, not 0
    auto poleend = (p+1+2*r-1).expect_int(); // ok, because np.array[0:3] eq array(1:3) in matlab
    auto wavebeg = (p+1+2*r-1).expect_int(); // p+1+2*r in matlab, because start index is 1, not 0

    int64_t L = (2+2*r+p+1).expect_int(); // number of params of a dilated, translated wavelet
    int64_t N = t.numel();
    int64_t n_ = n.expect_int();
    int64_t p_ = p.expect_int();
    int64_t r_ = r.expect_int();

    // Initialize Phi, dPhi and Ind
    auto Phi = torch::zeros({N, n_}, t.options());
    auto dPhi = torch::zeros({N, n_*L}, t.options());
    auto Ind = torch::zeros({2, n_*L}, t.options());
    auto dPhit = torch::zeros({N, n_}, t.options());

    // common parameters for all dilated, translated wavelets
    auto sigma = alpha[-1];
    auto ak    = alpha.slice(0, polebeg  , poleend-1, 2); // only real part of poles
    auto betak = alpha.slice(0, polebeg+1, poleend  , 2); // only imag part of poles
    auto pk    = alpha.slice(0, 0, p_);

    // Generate the wavelets and derivatives w.r.t. alpha
    for (int64_t k = 0; k < n_; k++)
    {
      // Break up alpha to make the code readable
      int64_t begindzers = k*L; // k*L+1 in matlab, because start index is 1, not 0
      int64_t endindzers = begindzers+p_; // p-1 in matlab

      int64_t begindpoles = k*L+polebeg;
      int64_t endindpoles = begindpoles+2*r_; // 2*r-1 in matlab

      int64_t begindsig = k*L+poleend+2;

      // Current dilation and translation
      auto s = alpha[wavebeg+2*k];
      auto ss = s_square? (at::pow(s,2) + smin) : s;
      auto x = alpha[wavebeg+2*k+1];
      auto tt = (t-x)/ss;

      auto [Psi, dPsix, dPsia, dPsib, dPsip, dPsiSigma] = psi_fun(tt, ak, betak, pk, bmin, {sigma.item()});

      auto sqrt_ss = torch::sqrt(ss);

      // Normalize Psi
      Psi = Psi / sqrt_ss;
      Phi.index_put_({Slice(), k}, at::real(Psi));
      dPsix = at::real(dPsix);
      dPsia = at::transpose(at::real(dPsia), 0, 1) / sqrt_ss;
      dPsib = at::transpose(at::real(dPsib), 0, 1) / sqrt_ss;
      dPsip = at::transpose(at::real(dPsip), 0, 1) / sqrt_ss;
      dPsiSigma = at::real(dPsiSigma) / sqrt_ss;
      dPhit.index_put_({Slice(), k}, dPsix / sqrt_ss);

      // Save derivatives and Ind values

      // zeros
      dPhi.index_put_({Slice(), Slice(begindzers, endindzers)}, dPsip);
      Ind.index_put_({0, Slice(begindzers, endindzers)}, k);
      Ind.index_put_({1, Slice(begindzers, endindzers)}, torch::arange(0, p));

      // poles
      dPhi.index_put_({Slice(), Slice(begindpoles, endindpoles-1, 2)}, dPsia);
      Ind.index_put_({0, Slice(begindpoles, endindpoles-1, 2)}, k);
      Ind.index_put_({1, Slice(begindpoles, endindpoles-1, 2)}, torch::arange(polebeg, poleend-1, 2));

      dPhi.index_put_({Slice(), Slice(begindpoles+1, endindpoles, 2)}, dPsib);
      Ind.index_put_({0, Slice(begindpoles+1, endindpoles, 2)}, k);
      Ind.index_put_({1, Slice(begindpoles+1, endindpoles, 2)}, torch::arange(polebeg+1, poleend, 2));

      auto invr_ss_1p5 = torch::pow(ss, -1.5);
      auto invr_ss_sqr = torch::pow(ss, -2.0);
      // Wavelet parameters
      auto dPsis = -0.5 * invr_ss_1p5 * Psi * sqrt_ss + (1.0 / sqrt_ss)  * dPsix * (-invr_ss_sqr) * (t - x).t();
      if (s_square) dPsis = dPsis * 2 * s;
      auto dPsit = dPsix * (-1.0) * invr_ss_1p5;

      int64_t begindwav = k * L + wavebeg;
      int64_t endindwav = begindwav + 2; // 1 in matlab

      dPsis = dPsis.unsqueeze(1);
      dPsit = dPsit.unsqueeze(1);

      dPhi.index_put_({Slice(), Slice(begindwav, endindwav)}, torch::cat({dPsis, dPsit}, 1));
      Ind.index_put_({0, Slice(begindwav, endindwav)}, k);
      Ind.index_put_({1, Slice(begindwav, endindwav)}, torch::arange(wavebeg + 2*k, wavebeg + 2*k + 2));

      // Sigma
      dPhi.index_put_({Slice(), begindsig}, dPsiSigma);
      Ind.index_put_({0, begindsig}, k);
      Ind.index_put_({1, begindsig}, (alpha.size(0) - 1));
    }
    
    Ind = Ind.to(torch::kInt64);

    return {Phi, dPhi, Ind, dPhit};
  }

  // Defines the operators
  TORCH_LIBRARY(torchwavelets_cpp, m)
  {
    m.def("_Q(Tensor x, Scalar a, Scalar beta, Scalar bmin) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_R(Tensor x, Scalar a, Scalar beta, Scalar bmin) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("psi_fun(Tensor x, Tensor ak, Tensor betak, Tensor pk, Scalar bmin, Scalar sigma) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("adaRatGaussWav(SymInt n, Tensor t, Tensor params, SymInt p, SymInt r, Scalar bmin, Scalar smin, bool s_square, ScalarType dtype, Device device) -> (Tensor, Tensor, Tensor, Tensor)");
  }

  // Registers CPU implementations
  TORCH_LIBRARY_IMPL(torchwavelets_cpp, CPU, m)
  {
    m.impl("_Q", &_Q);
    m.impl("_R", &_R);
    m.impl("psi_fun", &psi_fun);
    m.impl("adaRatGaussWav", &adaRatGaussWav);
  }
}
