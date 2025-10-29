import numpy as np
import torch

if __name__=="__main__":
    data = np.array([
        [[85,78],[67,82],[92,88],[75,70],[60,64]],
        [[70,68],[77,72],[85,90],[60,65],[78,76]],
        [[80,84],[88,87],[66,68],[72,73],[64,60]]
    ])

    #問題1
    torch_tensor = torch.tensor(data)
    print("==== probrem 1 ====")
    print(torch_tensor.size())

    #問題2
    permute_tensor = torch.permute(data)
    print("==== probrem 2 ====")
    print(permute_tensor)