import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn

import torchpoly_cpp
# reference Python implementation
import torchpoly_ref

NUM_RUNS = 100


def exec_for(num_runs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_runs): func(*args, **kwargs)
        return wrapper
    return decorator


class TestPolyFromRoots(TestCase):
    def sample_inputs(self):
        self.args = {
            "roots": torch.rand(25)
        }
    
    @exec_for(NUM_RUNS)
    def test_correctness_cpu(self):
        self.sample_inputs()
        res_cpp = torchpoly_cpp.poly_fromroots(**self.args)
        res_ref = torchpoly_ref.poly_fromroots(**self.args)
        torch.testing.assert_close(res_cpp, res_ref)


class TestPolyVal(TestCase):
    def sample_inputs(self):
        self.args = {
            "coeffs": torch.rand(25),
            "x": torch.rand(200),
        }
    
    @exec_for(NUM_RUNS)
    def test_correctness_cpu(self):
        self.sample_inputs()
        res_cpp = torchpoly_cpp.poly_val(**self.args)
        res_ref = torchpoly_ref.poly_val(**self.args)
        torch.testing.assert_close(res_cpp, res_ref)


class TestPolyDer(TestCase):
    def sample_inputs(self):
        self.args = {
            "coeffs": torch.rand(25)
        }
    
    @exec_for(NUM_RUNS)
    def test_correctness_cpu(self):
        self.sample_inputs()
        res_cpp = torchpoly_cpp.poly_der(**self.args)
        res_ref = torchpoly_ref.poly_der(**self.args)
        torch.testing.assert_close(res_cpp, res_ref)


class TestPsiFun(TestCase):
    def sample_inputs(self):
        self.args = {
            "x": torch.rand(20),
            "ak": torch.rand(5),
            "betak": torch.rand(10),
            "pk": torch.rand(8),
            "bmin": 2.0,
            "sigma": 5.0,
        }
    
    @exec_for(NUM_RUNS)
    def test_correctness_cpu(self):
        self.sample_inputs()
        res_cpp = torchpoly_cpp.psi_fun(**self.args)
        res_ref = torchpoly_ref.psi_fun(**self.args, device="cpu")
        for r_cpp, r_ref in zip(res_cpp, res_ref):
            torch.testing.assert_close(r_cpp, r_ref)


class TestAdaRatGaussWav(TestCase):
    def sample_inputs(self):
        self.args = {
            "n": 64,
            "t": torch.rand(1024),
            "params": torch.rand((2*64+11)+1),
            "p": 3,
            "r": 4,
            "bmin": 0.01,
            "smin": 0.01,
            "s_square": False,
            "dtype": torch.float,
            "device": "cpu"
        }

    @exec_for(NUM_RUNS)
    def test_correctness_cpu(self):
        self.sample_inputs()
        try:
            res_cpp = torchpoly_cpp.adaRatGaussWav(**self.args)
        except Exception as e: print('cpp>', e)
        try:
            res_ref = torchpoly_ref.adaRatGaussWav(**self.args)
        except Exception as e: print('ref>', e)
        for r_cpp, r_ref in zip(res_cpp, res_ref):
            torch.testing.assert_close(r_cpp, r_ref)


if __name__ == "__main__":
    unittest.main()
