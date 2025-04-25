from math import sqrt

import torch
from torch import nn

from torch_framework.spiking.models import AnalogNeuron


class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding="same",
                 kvco=500e6,
                 kpd=1/torch.pi,
                 timestep=1e-9,
                 threshold=-0.5,
                 alpha=0.5,
                 use_default_initialization=True,
                 mean_gain=0,
                 std_gain=1,
                 bias=False) -> None:
        super().__init__()

        self.core_layer = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bias=bias)
        self.scaling = kvco * kpd * timestep / 2
        self.threshold = threshold
        self.alpha = alpha

        if isinstance(kernel_size, int):
            in_neurons = in_channels * kernel_size
        else:
            in_neurons = in_channels
            for k in kernel_size:
                in_neurons *= k

        if not use_default_initialization:
            mean = mean_gain * threshold / in_neurons
            std = sqrt(std_gain / in_neurons)

            nn.init.uniform_(self.core_layer.weight, mean-std, mean+std)

        self.neuron = AnalogNeuron.apply

    def forward(self, inputs):
        outs = []
        for step in range(inputs.shape[-1]):
            out = self.core_layer(inputs[..., step])
            outs.append(out)
        outs = torch.stack(outs, dim=-1)

        if not self.training:
            neuron_constant = (1 + self.core_layer.weight.sum((1, 2, 3))) / self.scaling
            neuron_constant.clamp_(0)
            neuron_constant = neuron_constant.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        else:
            neuron_constant = 1 / self.scaling

        scale = -1 / (1 + neuron_constant)
        mult = (neuron_constant - 1) / (neuron_constant + 1)

        return self.neuron(outs, self.threshold, scale, mult, self.alpha)
