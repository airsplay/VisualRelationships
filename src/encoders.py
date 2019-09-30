from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
from param import args


class DiffModule(nn.Module):
    """
    Trying to find the difference between the two input for a siliency map
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = LinearAct(input_dim, output_dim)
        self.fc2 = LinearAct(input_dim, output_dim)
        self.fc3 = nn.Sequential(
            LinearAct(output_dim * 2, output_dim, 'relu'),
            LinearAct(output_dim, output_dim)
        )

    def forward(self, src, trg):
        """
        :param src: B, l, d
        :param trg: B, n, d
        :return:
        """
        src_value = self.fc1(src)       # B, l, h
        trg_key = self.fc2(trg)         # B, n, h
        trg_value = self.fc1(trg)       # B, n, h

        score = torch.einsum("ijd,ikd->ijk", src_value, trg_key) / math.sqrt(self.output_dim)
        prob = F.softmax(score, -1)     # B, l, n

        src_ctx = torch.matmul(prob, trg_value)     # B, l, n * B, n, h --> B, l, h

        return self.fc3(torch.cat((src_value, src_value - src_ctx), -1))





