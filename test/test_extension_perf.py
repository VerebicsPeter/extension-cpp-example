import timeit
import torch
import torchpoly_cpp
import torchpoly_ref

N = 6#64
P = 3#15
R = 4#20

args = lambda n,p,r,t : {
    "n": n,
    "t": torch.rand(t),
    "params": torch.rand((2*n+p+2*r)+1),
    "p": p,
    "r": r,
    "bmin": 0.01,
    "smin": 0.01,
    "s_square": False,
    "dtype": torch.float,
    "device": "cpu"
}

def test_perf(n,p,r,t=1024,number=100):
    cpp_time = timeit.timeit(lambda : torchpoly_cpp.adaRatGaussWav(**args(n,p,r,t)), number=number)
    print("cpp done.")
    ref_time = timeit.timeit(lambda : torchpoly_ref.adaRatGaussWav(**args(n,p,r,t)), number=number)
    print("ref done.")
    print("cpp time:", cpp_time)
    print("ref time:", ref_time)
    print("rate:", ref_time / cpp_time)


if __name__ == "__main__":
    print("\nTesting performance on small inputs...")
    test_perf(n=6, p=3, r=4, number=100)
    print("\nTesting performance on large inputs...")
    test_perf(n=64, p=15, r=20, number=10)
