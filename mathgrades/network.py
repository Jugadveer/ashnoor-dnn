"""The neural network, written as a PyTorch nn.Module."""

import torch
from torch import nn
from torch.nn.functional import softplus


class OrderedThresholds(nn.Module):
    """K-1 biases that always decrease: b_1, b_1 - softplus(d_1), ...

    softplus(d) = log(1 + e^d) is always positive, so every step down is
    positive whatever the network learns. Used by the ordinal head.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.first = nn.Parameter(torch.zeros(1))
        self.gaps = nn.Parameter(torch.zeros(n_classes - 2))

    def forward(self) -> torch.Tensor:
        steps = torch.cat([torch.zeros(1), softplus(self.gaps)])
        return self.first - torch.cumsum(steps, dim=0)


class GradeNet(nn.Module):
    """A feed-forward network with an ordinal output and a linear skip path.

    Hidden layers: each is Linear -> BatchNorm1d -> ReLU -> Dropout.
    - Linear: weighted sum of the inputs plus a bias.
    - BatchNorm1d: rescales each neuron's output over the mini-batch to mean 0
      and variance 1, then applies a learnt scale and shift; keeps values in a
      stable range so training is faster and less sensitive to the learning rate.
    - ReLU: max(0, x), the non-linearity that lets the network learn curves.
    - Dropout: zeroes each neuron with probability p during training so the
      network cannot lean on any single neuron (fights overfitting).

    Ordinal head (head="ordinal", CORAL-style, Cao, Mirjalili and Raschka 2020):
    the network computes ONE ability score per student,
        score = w . hidden + v . inputs       (v . inputs is the linear skip path)
    and K-1 = 3 threshold logits  score + b_k  answer "is the band above
    At risk / above Average / above Good?". Sharing the score and keeping the
    biases ordered guarantees P(above At risk) >= P(above Average) >= P(above Good).

    The linear skip path lets the network copy a straight-line rule such as
    "final grade follows the second-term grade" directly, so the hidden layers
    only have to learn the corrections. linear_skip=False removes it (ablation).

    head="softmax" is the earlier version (K band logits), kept for the ablation.
    """

    def __init__(self, n_inputs: int, hidden_sizes: tuple[int, ...], dropout: float,
                 n_classes: int, batch_norm: bool = True, head: str = "ordinal",
                 linear_skip: bool = True):
        super().__init__()
        blocks = []
        width_in = n_inputs
        for width in hidden_sizes:
            blocks.append(nn.Linear(width_in, width))
            if batch_norm:
                blocks.append(nn.BatchNorm1d(width))
            blocks += [nn.ReLU(), nn.Dropout(dropout)]
            width_in = width
        self.hidden = nn.Sequential(*blocks)

        self.ordinal = head == "ordinal"
        out_width = 1 if self.ordinal else n_classes
        self.output = nn.Linear(width_in, out_width, bias=not self.ordinal)
        self.skip = nn.Linear(n_inputs, out_width, bias=False) if linear_skip else None
        self.thresholds = OrderedThresholds(n_classes) if self.ordinal else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.output(self.hidden(x))
        if self.skip is not None:
            out = out + self.skip(x)
        if self.ordinal:
            out = out + self.thresholds()  # (students, 1) + (K-1,) -> (students, K-1)
        return out
