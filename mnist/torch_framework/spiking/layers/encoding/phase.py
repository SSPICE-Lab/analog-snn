"""
Module containing the implementation of the `PhaseEncoding` class.
"""

import numpy as np
import torch


class PhaseEncoding(torch.nn.Module):
    """
    Class implementing phase encoding of inputs
    """

    def __init__(self,
                 timesteps: int,
                 prescale: float = 1 / 128,
                 repeat_scale: float = 1 / 256
        ) -> None:
        """
        Initialize the `PhaseEncoding` class.

        Parameters
        ----------
        timesteps : int
            Number of timesteps to encode to

        prescale : float, optional
            Constant to multiply the encoded vector with
            Defaults to `1 / 128`.

        repeat_scale : float, optional
            Constant to scale repetitions of the encoded vector with
            Defaults to `1 / 256`.
        """

        super().__init__()

        self.timesteps = timesteps
        self.n_repeats = int(np.ceil(timesteps / 8))
        self.register_buffer("bit_array",
                             torch.Tensor(1 << np.arange(8)[::-1]).int(),
                             persistent=False)
        self.prescale = prescale
        self.repeat_scale = repeat_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = (inputs * 255).int()
        encoded_input = torch.bitwise_and(inputs[..., None], self.bit_array).float() * self.prescale

        ret = []
        for _ in range(self.n_repeats):
            ret.append(encoded_input)
            encoded_input = encoded_input.clone() * self.repeat_scale
        ret = torch.cat(ret, dim=-1)
        ret.requires_grad = False

        return ret[..., :self.timesteps].to(dtype=torch.float32)
