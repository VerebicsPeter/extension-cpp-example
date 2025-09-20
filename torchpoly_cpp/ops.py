import torch
from torch import Tensor

__all__ = ["poly_mul", "poly_fromroots", "poly_val", "poly_der"]

# Example for torch docs
'''
def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)
'''


def poly_mul(p1: Tensor, p2: Tensor) -> Tensor:
    return torch.ops.torchpoly_cpp.poly_mul.default(p1, p2)

@torch.library.register_fake("torchpoly_cpp::poly_mul")
def _(p1, p2):
    torch._check(p1.dtype == torch.float)
    torch._check(p2.dtype == torch.float)
    torch._check(p1.device == p2.device)
    size = p1.size(0) + p2.size(0) - 1
    return torch.empty(size, dtype=torch.float, device=p1.device)


def poly_fromroots(roots: Tensor) -> Tensor:
    return torch.ops.torchpoly_cpp.poly_fromroots.default(roots)

@torch.library.register_fake("torchpoly_cpp::poly_fromroots")
def _(roots):
    torch._check(roots.dtype == torch.float)
    size = roots.size(0) + 1
    return torch.empty(size, dtype=torch.float, device=roots.device)


def poly_val(coeffs: Tensor, x: Tensor | float) -> Tensor:
    if isinstance(x, Tensor):
        if x.dim() != 0:
            raise ValueError("'poly_val' only supports scalar values, not tensors.")
        x = x.item()

    return torch.ops.torchpoly_cpp.poly_val.default(coeffs, x)

@torch.library.register_fake("torchpoly_cpp::poly_val")
def _(coeffs, _):
    torch._check(coeffs.dtype == torch.float)
    size = ()
    return torch.empty(size, dtype=torch.float, device=coeffs.device)


def poly_der(coeffs: Tensor) -> Tensor:
    return torch.ops.torchpoly_cpp.poly_der.default(coeffs)

@torch.library.register_fake("torchpoly_cpp::poly_der")
def _(coeffs):
    torch._check(coeffs.dtype == torch.float)
    size = coeffs.size(0) - 1
    return torch.empty(size, dtype=torch.float, device=coeffs.device)
