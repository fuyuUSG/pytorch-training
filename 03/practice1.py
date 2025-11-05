import torch
from torch import nn

if __name__ == "__main__":
    my_tensor = torch.ones((32, 3, 128, 128))
    print(f"practice1: {my_tensor.shape}")

    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    out = conv(my_tensor)
    print(f"practice2: {out.shape}")

    conv2 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1)
    out2 = conv2(my_tensor)
    print(f"practice3: {out2.shape}")

    conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
    out3 = conv3(my_tensor)
    print(f"practice4_2: {out3.shape}")

    conv4 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=2, padding=2)
    out4 = conv4(my_tensor)
    print(f"practice4_2 : {out4.shape}")