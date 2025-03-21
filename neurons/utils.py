import torch

class ThresholdBelow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        ctx.set_materialize_grads(False)

        return (inputs < threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            return grad_input, None, None

        return None, None, None

class ThresholdAbove(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        ctx.set_materialize_grads(False)

        return (inputs > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            return grad_input, None, None

        return None, None, None
