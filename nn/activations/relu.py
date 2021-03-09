import nn.module.Module as Module
from zensor import Zensor as Z
from zensor.Zensor import zRelu


class ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Z) -> Z:
        return zRelu(x)
