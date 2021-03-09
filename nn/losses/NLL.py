import nn.module.Module as Module
from zensor import Zensor as Z

class NLL(Module):
    def __init__(self):
        pass

    def forward(self, x:Z, target)->Z:
        return -x[range(target.value.shape[0]), target.value].mean()
