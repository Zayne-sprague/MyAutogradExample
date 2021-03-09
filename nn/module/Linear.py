import nn.module.Module as Module
from zensor import Zensor as Z
import numpy as np


class Linear(Module):

    def __init__(self, fan_in, fan_out, use_bias=True):
        super().__init__()

        self.weights = Z(np.random.random([fan_in, fan_out]) / np.sqrt(fan_in))
        self.bias = Z(np.ones([fan_out])) if use_bias else None


    def forward(self, x, *args, **kwargs):
        if self.bias:
            return x @ self.weights + self.bias

        return x @ self.weights




if __name__ == "__main__":
    import torch
    from nn.activations.sigmoid import Sigmoid

    A = np.random.random([3,10])
    B = np.random.random([10, 20])

    t_x = torch.tensor(A.copy(), requires_grad=True)
    t_w = torch.tensor(B.copy(), requires_grad=True)
    t_y = (t_x @ t_w).sum()

    t_x.retain_grad()
    t_w.retain_grad()
    t_y.retain_grad()

    t_y.backward()


    x = Z(A.copy())

    layer = Linear(10, 20, True)
    layer.weights = Z(B.copy())

    act = Sigmoid()
    y = act(layer(x))

    y = y.sum()

    y.backward()
    print(y)
    print("-")




