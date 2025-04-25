from math import sqrt

import torch
from torch import nn

from torch_framework.spiking.models import AnalogNeuron


class Dense(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kvco=500e6,
                 kpd=1/torch.pi,
                 timestep=1e-9,
                 threshold=-0.5,
                 alpha=0.5,
                 use_default_initialization=True,
                 mean_gain=0,
                 std_gain=1) -> None:
        super().__init__()

        self.core_layer = nn.Linear(in_features, out_features, bias=False)
        self.scaling = kvco * kpd * timestep / 2
        self.threshold = threshold
        self.alpha = alpha

        if not use_default_initialization:
            mean = mean_gain * threshold / in_features
            std = sqrt(std_gain / in_features)

            nn.init.uniform_(self.core_layer.weight, mean-std, mean+std)

        self.neuron = AnalogNeuron.apply

    def forward(self, inputs):
        outs = []
        for step in range(inputs.shape[-1]):
            out = self.core_layer(inputs[..., step])
            outs.append(out)
        outs = torch.stack(outs, dim=-1)

        neuron_constant = (1 + self.core_layer.weight.sum(1)) / self.scaling
        neuron_constant.clamp_(0)
        neuron_constant = neuron_constant.unsqueeze(0)

        scale = -1 / (1 + neuron_constant)
        mult = (neuron_constant - 1) / (neuron_constant + 1)

        return self.neuron(outs, self.threshold, scale, mult, self.alpha)
