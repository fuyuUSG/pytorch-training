import torch
from torch import nn

if __name__ == "__main__":
    in_tensor = torch.ones(32, 3, 128, 128)

    print(f"practice1: {in_tensor.shape}")

    conv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=8)
    out = conv(in_tensor)
    
    norm = nn.BatchNorm2d(num_features=256)
    out1 = norm(out)

    relu = nn.ReLU()
    out2 = relu(out1)
    out2_flat = out2.view(out2.shape[0], -1)

    fc = nn.Linear(in_features=256*16*16, out_features=126, bias=True)
    out3 = fc(out2_flat)
    print(f"practice1: {out3.shape}")