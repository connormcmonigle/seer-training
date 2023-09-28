import torch
import torch.autograd as autograd

class InjectLassoLoss(autograd.Function):
    FT_REGULARIZATION = 1 / 4194304 / 4096

    @staticmethod
    def setup_context(ctx, inputs, output):
        input,  = inputs
        ctx.save_for_backward(input)

    @staticmethod
    def forward(input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return InjectLassoLoss.FT_REGULARIZATION * torch.sign(input) + grad_output

inject_lasso_loss = InjectLassoLoss.apply