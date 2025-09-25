import torch
from pathlib import Path
from . import _C, ops

from .ops import (
    poly_mul,
    poly_fromroots, 
    poly_val,
    poly_der,
    psi_fun,
    adaRatGaussWav,
)
