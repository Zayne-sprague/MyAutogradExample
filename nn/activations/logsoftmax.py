import nn.module.Module as Module
from zensor import Zensor as Z
import numpy as np


class LogSoftmax(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Z) -> Z:
        # return x - x.exp().sum(-1).log().unsqueeze(-1)

        mx = np.max(x.value)
        logsumexp = (((x - mx).exp()).sum()).log()
        return x - mx - logsumexp

        # z = x - np.max(x.value)
        # numerator = z.exp()
        # denominator = numerator.sum()
        # softmax = numerator / denominator
        # return softmax.log()
