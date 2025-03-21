import torch
from . import utils

class RO_Dense(torch.nn.Module):
    def __init__(self, in_neurons, out_neurons, kvco, ffree, vth, std_gain=None):
        super(RO_Dense, self).__init__()
        self.const = ffree / kvco
        self.core_layer = torch.nn.Linear(in_neurons, out_neurons, bias=False)
        self.vth = vth
        self.comparator = utils.ThresholdBelow.apply

        if std_gain is not None:
            self.core_layer.weight.data *= std_gain

    @property
    def weight(self):
        return self.core_layer.weight

    @property
    def bias(self):
        return self.core_layer.bias

    def forward(self, x):
        batch_size = x.shape[0]
        timesteps = x.shape[1]
        x = x.reshape(batch_size * timesteps, x.shape[-1])
        x = self.core_layer(x)
        x = x.reshape(batch_size, timesteps, x.shape[-1])

        # Neuron model
        const = self.const * (1 + torch.sum(torch.abs(self.core_layer.weight.detach()), dim=1))
        spikes = torch.zeros_like(x)
        for t in range(timesteps):
            if t == 0:
                prev_x = torch.zeros(batch_size, x.shape[-1])
                prev_vm = torch.zeros(batch_size, x.shape[-1])
            x_now = x[:, t, :]
            vm = -(prev_vm * (1 - const) + x_now + prev_x) / (1 + const)
            spikes[:, t, :] = self.comparator(vm, self.vth)
            prev_x = x_now * (1 - spikes[:, t, :].detach())
            prev_vm = vm * (1 - spikes[:, t, :].detach())

        return spikes
