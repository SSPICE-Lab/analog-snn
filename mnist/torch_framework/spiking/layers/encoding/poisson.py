"""
Module containing the implementation of the `PoissonEncoding` class.
"""

import numpy as np
import torch


class PoissonEncoding(torch.nn.Module):
    """
    Class implementing Poisson encoding of inputs
    """

    def __init__(self,
                 timesteps: int,
                 lambda_: float = 1.0,
        ) -> None:
        """
        Initialize the `PoissonEncoding` class.

        Parameters
        ----------
        timesteps : int
            Number of timesteps to encode to
        """

        super().__init__()

        self.timesteps = timesteps
        self.lambda_ = lambda_

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Inputs shape: [batch_size, num_channels, height, width]
        # Outputs shape: [batch_size, num_channels, height, width, timesteps]
        random_numbers = torch.rand(inputs.shape + (self.timesteps,), device=inputs.device)
        return (random_numbers < (inputs[..., None] / self.lambda_)).float()
