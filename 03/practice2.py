import torch
from torch import nn

if __name__ == "__main__":
    _in = torch.ones((32, 1024))
    print(f"_in: {_in.shape}")

    fc = nn.Linear(in_features=1024, out_features=256, bias=True)
    out = fc(_in)
    print(f"practice1: {out.shape}")

    fc2 = nn.Linear(in_features=1024, out_features=2048, bias=True)
    out2 = fc2(_in)
    print(f"practice2: {out2.shape}")

    out3 = out.reshape(32, 16, 16)
    print(f"practice3: {out3.shape}")