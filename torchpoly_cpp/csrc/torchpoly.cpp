#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace torchpoly_cpp {

/*
def poly_mul(p1, p2):
    degree1 = p1.size(0) - 1 # n
    degree2 = p2.size(0) - 1 # m
    
    result_degree = degree1 + degree2 # n+m
    result = torch.zeros(result_degree + 1, dtype=p1.dtype) # n+m+1
    
    for i in range(p1.size(0)):
        for j in range(p2.size(0)):
            result[i + j] += p1[i] * p2[j]
    
    return result
*/
at::Tensor poly_mul(const at::Tensor& p1, const at::Tensor& p2) {
  TORCH_CHECK(p1.dtype() == at::kFloat);
  TORCH_CHECK(p2.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(p1.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(p2.device().type() == at::DeviceType::CPU);
  at::Tensor p1_contig = p1.contiguous();
  at::Tensor p2_contig = p2.contiguous();
                                  // n+1             +  m+1 - 1 = n+m+1
  at::Tensor result = torch::zeros({p1_contig.size(0)+p2_contig.size(0)-1,}, p1_contig.options());
  
  const float* p1_ptr = p1_contig.data_ptr<float>();
  const float* p2_ptr = p2_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  for (int64_t i = 0; i < p1_contig.numel(); i++) {
    for (int64_t j = 0; j < p2_contig.numel(); j++) {
      result_ptr[i + j] += p1_ptr[i] * p2_ptr[j];
    }
  }
  return result;
}

/*
# highest degree first (not numpy equvivalent: lowest degree first)
def torch_polyfromroots(roots):
    n = roots.shape[0]

    coeffs = torch.zeros((n,))
    for k in range(n):
        if k == 0:
            Q = torch.tensor([1, -roots[0]])
            coeffs = Q
        else:
            Q = torch.tensor([1, -roots[k]])
            coeffs = poly_mul(Q, coeffs)
    return coeffs
*/
at::Tensor poly_fromroots(const at::Tensor& roots) {
  TORCH_CHECK(roots.dtype() == at::kFloat);
  TORCH_CHECK(roots.device().type() == at::DeviceType::CPU);
  
  at::Tensor roots_contig = roots.contiguous();
  const float* roots_ptr = roots_contig.data_ptr<float>();
  
  at::Tensor coeffs;
  at::Tensor Q;
  for (int64_t k = 0; k < roots_contig.numel(); k++) {
    Q = torch::tensor({1.0f, -roots_ptr[k]}, roots_contig.options());
    if (k==0) {
      coeffs = Q;
    } else {
      coeffs = poly_mul(Q, coeffs);
    }
  }
  return coeffs;
}

/*
# expects coeffs highest degree first
def torch_polyval(coeffs, x):
    """Evaluate a polynomial at given x using Horner's method."""
    # x = torch.tensor(x, dtype=coeffs.dtype, device=coeffs.device)  # Ensure x is on the same device
    result = torch.zeros_like(x)
    
    for c in coeffs:
        result = result * x + c

    return result
*/
at::Tensor poly_val(const at::Tensor& coeffs, const at::Scalar& x) {
  TORCH_CHECK(coeffs.dtype() == at::kFloat);
  TORCH_CHECK(coeffs.device().type() == at::DeviceType::CPU);
  
  at::Tensor coeffs_contig = coeffs.contiguous();
  const float* coeffs_ptr = coeffs_contig.data_ptr<float>();

  float rf = 0.0f;
  // NOTE: convert scalar to float 
  // (float arg is not allowed for passing scalars)
  float xf = x.to<float>();
  for (int64_t i = 0; i < coeffs_contig.numel(); i++) {
    rf = rf * xf + coeffs_ptr[i];
  }

  at::Tensor result = torch::tensor({rf}, coeffs_contig.options()).squeeze();
  return result;
}

/*
def torch_polyder(coeffs):
    """Compute the derivative of a polynomial given its coefficients."""
    order = torch.arange(len(coeffs) - 1, 0, -1, dtype=coeffs.dtype, device=coeffs.device)
    return coeffs[:-1] * order  # Multiply by the respective exponent
*/
at::Tensor poly_der(const at::Tensor& coeffs) {
  TORCH_CHECK(coeffs.dtype() == at::kFloat);
  TORCH_CHECK(coeffs.device().type() == at::DeviceType::CPU);
  int64_t n = coeffs.size(0) - 1;
  at::Tensor order = torch::arange(n, 0, -1, coeffs.options()).to(at::kFloat);
  at::Tensor slice = coeffs.slice(0, 0, n);
  return slice * order;
}

// Defines the operators
TORCH_LIBRARY(torchpoly_cpp, m) {
  m.def("poly_mul(Tensor p1, Tensor p2) -> Tensor");
  m.def("poly_fromroots(Tensor roots) -> Tensor");
  m.def("poly_val(Tensor coeffs, Scalar x) -> Tensor");
  m.def("poly_der(Tensor coeffs) -> Tensor");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(torchpoly_cpp, CPU, m) {
  m.impl("poly_mul", &poly_mul);
  m.impl("poly_fromroots", &poly_fromroots);
  m.impl("poly_val", &poly_val);
  m.impl("poly_der", &poly_der);
}

}
