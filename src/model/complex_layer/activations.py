# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, x):
        # Apply ReLU to both real and imaginary parts
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexSiLU(nn.Module):
    def __init__(self):
        super(ComplexSiLU, self).__init__()

    def forward(self, x):
        # Apply SiLU (Swish) to both real and imaginary parts
        return torch.complex(F.silu(x.real), F.silu(x.imag))


class ComplexGeLU(nn.Module):
    def __init__(self):
        super(ComplexGeLU, self).__init__()

    def forward(self, x):
        # Apply GeLU to both real and imaginary parts
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))


class ComplexTanH(nn.Module):
    def __init__(self):
        super(ComplexTanH, self).__init__()

    def forward(self, x):
        exp_z = torch.exp(x)
        exp_neg_z = torch.exp(-x)

        return (exp_z - exp_neg_z) / (exp_z + exp_neg_z + 1e-9)
