"""
Module containing the implementation of the `Sequential` class
"""

from typing import Union

import torch
from torch import nn

from torch_framework.network import Network


def _build_conv2d(num_filters, input_shape, **kwargs):
    kernel_size = kwargs.get("kernel_size", (3, 3))
    stride = kwargs.get("stride", (1, 1))
    padding = kwargs.get("padding", "same")
    bias = kwargs.get("bias", True)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    if len(input_shape) == 4:
        batch_size = input_shape[0]
    else:
        batch_size = 1
    x_dim = input_shape[1]
    y_dim = input_shape[2]

    if padding == "valid":
        x_dim -= kernel_size[0] // 2
        y_dim -= kernel_size[1] // 2

    x_dim = x_dim // stride[0]
    y_dim = y_dim // stride[1]

    out_shape = [batch_size, x_dim, y_dim, num_filters]

    layer = nn.Conv2d(input_shape[3], num_filters, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias)

    return layer, out_shape


class Sequential(Network):
    """
    `Sequential` network class
    """

    def __init__(self, input_shape: list) -> None:
        """
        Initialize the `Sequential` network

        Parameters
        ----------
        input_shape : list
            Shape of the input to the network
            Expects a list with first element to be `batch_size`
        """

        super().__init__()

        self.layers = nn.ModuleList()
        self.input_shape = list(input_shape)
        self.current_shape = list(input_shape)

    def add(self,
            layer: Union[str, nn.Module],
            *args,
            **kwargs):
        """
        Appends a layer to the list
        Input shape of the layer is computed automatically

        Parameters
        ----------
        layer : str | torch.nn.Module
            Layer to be added
            Currently supported layers: ["conv2d", "maxpool2d", "relu", "flatten", "linear", "dropout"]
            Can also pass a torch layer
        """

        layer_list = ["conv2d", "maxpool2d", "relu", "flatten", "linear", "dropout"]

        if isinstance(layer, nn.Module):
            self.layers.append(layer)
            return

        if not layer.lower() in layer_list:
            raise ValueError(f"Layer `{layer}` not supported")

        layer = layer.lower()
        if layer == "conv2d":
            if len(args) == 0:
                if not "num_filters" in kwargs.keys():
                    raise ValueError("`num_filters` kwarg necessary for layer `conv2d`")
                num_filters = kwargs["num_filters"]
                kwargs.pop("num_filters")
            else:
                num_filters = args[0]

            built_layer, out_shape = _build_conv2d(num_filters, self.current_shape, **kwargs)

            self.current_shape = out_shape
            self.layers.append(built_layer)

        elif layer == "maxpool2d":
            kernel_size = kwargs.get("kernel_size", (2, 2))
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            stride = kwargs.get("stride", kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)

            self.current_shape[1] = 1 + (self.current_shape[1] - kernel_size[0]) // stride[0]
            self.current_shape[2] = 1 + (self.current_shape[2] - kernel_size[1]) // stride[1]

            self.layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

        elif layer == "relu":
            self.layers.append(nn.ReLU())

        elif layer == "flatten":
            self.current_shape = [self.current_shape[0], self.current_shape[1]
                                  * self.current_shape[2] * self.current_shape[3]]

            self.layers.append(nn.Flatten())

        elif layer == "linear":
            if len(args) == 0:
                if not "units" in kwargs.keys():
                    raise ValueError("`units` kwarg necessary for layer `linear`")
                units = kwargs["units"]
            else:
                units = args[0]

            bias = kwargs.get("bias", True)

            self.layers.append(nn.Linear(self.current_shape[1], units, bias=bias))
            self.current_shape[1] = units

        elif layer == "dropout":
            if len(args) == 0:
                p = kwargs.get("p", 0.5)
            else:
                p = args[0]

            self.layers.append(nn.Dropout(p))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        """

        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
