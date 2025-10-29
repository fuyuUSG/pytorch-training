import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, mytensor: torch.Tensor, elem_add, elem_multiply):
        super().__init__()
        self.mytensor = mytensor
        self.elem_add = elem_add
        self.elem_multiply = elem_multiply

    def forward(self, x):
        prob2 = x + self.mytensor
        prob3 = prob2 + elem_add
        prob4 = prob3 * elem_multiply

if __name__ == "__main__":
    mymodel = MyModel(3,2.0)
    _input = torch.ones([2,3])
    print(prob2)
    print(prob3)
    print(prob4)