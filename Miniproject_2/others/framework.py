from torch import empty, cat, arange
from torch.nn.functional import fold, unfold


class Module:
    def __init__(self):
        # Initialize the module
        pass

    def forward(self, *inputs):
        # Forward pass
        raise NotImplementedError

    def backward(self, grad):
        # Backward pass
        raise NotImplementedError

    def param(self):
        # Return the parameters of the module
        return []

    def zero_grad(self):
        # Set the gradients to zero
        for _, gradp in self.param():
            gradp.zero_()

    def cuda(self):
        # Move the module parameters to CUDA
        if len(self.param()) > 0:
            self.weight = self.weight.to(device='cuda')
            self.bias = self.bias.to(device='cuda')
            self.gradw = self.gradw.to(device='cuda')
            self.gradb = self.gradb.to(device='cuda')

    def __call__(self, *inputs):
        return self.forward(*inputs)

class ReLU(Module):
    def forward(self, input):
        self.input = input
        return self.input.relu()

    def backward(self, grad):
        gradf = self.input.heaviside(
            values=empty(
                1,
                dtype=self.input.dtype,
                device=self.input.device,
            ).zero_()
            )
        return grad * gradf

class Sigmoid(Module):
    def forward(self, input):
        self.input = input
        return self.input.sigmoid()

    def backward(self, grad):
        gradf = self.input.sigmoid() * (1 - self.input.sigmoid())
        return grad * gradf

class MSE(Module):
    def forward(self, input, target):
        self.input, self.target = input, target
        return (self.input - self.target).pow(2).mean()

    def backward(self, grad=None):
        # If no grad is given, initialize it with ones
        if grad is None:
            grad = empty(1, device=self.input.device).fill_(1.)
        # Calculate gradient of the output with respect to the input
        gradf = 2 * (self.input - self.target) / self.input.numel()
        # Return the gradient with respect to the input and the target
        return grad * gradf, grad * (-gradf)

class Sequential(Module):
    def __init__(self, *modules):
        # Store the modules
        self.modules = modules

    def forward(self, input):
        # Pass the output of every layer to its next layer
        output = input.clone()
        for module in self.modules:
            output = module(output)
        return output

    def backward(self, grad):
        # Propagate the grad of each layer to its previous layer
        for module in self.modules[::-1]:
            grad = module.backward(grad)
        return grad

    def cuda(self):
        # Call the CUDA method of each module
        for module in self.modules:
            module.cuda()

    def zero_grad(self):
        # Call the zero_grad method of each module
        for module in self.modules:
            module.zero_grad()

    def param(self):
        # Return the parameters of all modules
        param = []
        for module in self.modules:
            param += module.param()
        return param

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), params=None):

        # Check the parameters
        for param in [kernel_size, stride, padding, dilation]:
            if type(param) not in [tuple, int]:
                raise TypeError('The parameter must be either an int or a tuple.')

        # Store the settings
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.settings = dict(
            kernel_size=kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size),
            stride=stride if isinstance(stride, tuple) else (stride, stride),
            padding=padding if isinstance(padding, tuple) else (padding, padding),
            dilation=dilation if isinstance(dilation, tuple) else (dilation, dilation),
            )
        kernel_size = self.settings['kernel_size']

        # Get or generate the parameters
        sqrtk = (1 / (kernel_size[0] * kernel_size[1] * in_channels)) ** .5
        if params:
            weight = params['weight']
            bias = params['bias']
            assert weight.shape == (out_channels, in_channels, *kernel_size,)
            assert bias.shape == (out_channels,)
        else:
            weight = empty(
                out_channels, in_channels, *kernel_size).uniform_(-sqrtk, sqrtk)
            bias = empty(
                out_channels).uniform_(-sqrtk, sqrtk)

        # Store the parameters and their gradients
        self.weight = weight
        self.bias = bias
        self.gradw = empty(
            weight.shape, dtype=weight.dtype, device=weight.device).zero_()
        self.gradb = empty(
            bias.shape, dtype=bias.dtype, device=bias.device).zero_()

    def forward(self, input):
        # Get the settings
        kernel_size = self.settings['kernel_size']
        stride = self.settings['stride']
        padding = self.settings['padding']
        dilation = self.settings['dilation']

        # Reshape the parameters
        weight_ = self.weight.reshape(self.out_channels, -1)
        bias_ = self.bias.reshape(1, -1, 1)

        # Calculate the output
        input_ = unfold(input, **self.settings).to(input.device)
        output_ = (weight_ @ input_ + bias_)
        output = output_.reshape(
            input.shape[0], self.out_channels,
            int(1 + (input.shape[2]  + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]),
            int(1 + (input.shape[3]  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]),
            )

        # Store the input and the output for the backward propagation
        self.input = input
        self.output = output

        return output

    def backward(self, grad):
        # Check the input shape
        assert grad.shape == self.output.shape

        # Get the settings
        kernel_size = self.settings['kernel_size']

        # Reshape the weight
        weight_ = self.weight.reshape(self.out_channels, -1)

        # Calculate the gradient wrt the input
        grad_ = grad.reshape(grad.shape[0], grad.shape[1], -1)
        gradx_ = weight_.transpose(0, 1) @ grad_
        gradx = fold(
            gradx_,
            output_size=(self.input.shape[2], self.input.shape[3]), **self.settings
            )
        assert gradx.shape == self.input.shape

        # Calculate the gradient wrt the weight
        input_ = unfold(self.input, **self.settings)
        gradw_ = (grad_ @ input_.transpose(1, 2)).sum(dim=0)

        # Accumulate the gradients wrt the parameter
        self.gradw += gradw_.reshape(self.out_channels, self.in_channels, *kernel_size)
        self.gradb += grad_.sum(dim=(0, 2))

        return gradx

    def param(self):
        return [
            (self.weight, self.gradw),
            (self.bias, self.gradb),
            ]

class TransposeConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), params=None):

        # Check the parameters
        for param in [kernel_size, stride, padding, dilation]:
            if type(param) not in [tuple, int]:
                raise TypeError('The parameter must be either an int or a tuple.')
        if dilation not in [(1, 1), 1]:
            raise ValueError('The dilation must be either (1, 1) or 1. Other dilations are not supported.')

        # Store the settings
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.settings = dict(
            kernel_size=kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size),
            stride=stride if isinstance(stride, tuple) else (stride, stride),
            padding=padding if isinstance(padding, tuple) else (padding, padding),
            dilation=dilation if isinstance(dilation, tuple) else (dilation, dilation),
            )
        kernel_size = self.settings['kernel_size']

        # Get or generate the parameters
        sqrtk = (1 / (kernel_size[0] * kernel_size[1] * in_channels)) ** .5
        if params:
            weight = params['weight']
            bias = params['bias']
            assert weight.shape == (in_channels, out_channels, *kernel_size,)
            assert bias.shape == (out_channels,)
        else:
            weight = empty(
                in_channels, out_channels, *kernel_size).uniform_(-sqrtk, sqrtk)
            bias = empty(
                out_channels).uniform_(-sqrtk, sqrtk)

        # Store the parameters
        self.weight = weight
        self.bias = bias
        self.gradw = empty(
            weight.shape, dtype=weight.dtype, device=weight.device).zero_()
        self.gradb = empty(
            bias.shape, dtype=bias.dtype, device=bias.device).zero_()

    def forward(self, input):
        # Get the settings
        kernel_size = self.settings['kernel_size']
        stride = self.settings['stride']
        padding = self.settings['padding']
        dilation = self.settings['dilation']

        # Reshape the parameters
        weight_ = self.weight.transpose(0, 1).flip(dims=(2, 3,)).reshape(self.out_channels, -1)
        bias_ = self.bias.reshape(1, -1, 1)

        # Calculate the output
        input_ = unfold(
            self.dilate(input, dilation=stride),
            kernel_size=kernel_size,
            padding=(kernel_size[0] - 1, kernel_size[1] - 1)
            )
        output_ = (weight_ @ input_ + bias_)
        output = output_.reshape(
            input.shape[0],
            self.out_channels,
            stride[0] * (input.shape[2] - 1) + kernel_size[0],
            stride[1] * (input.shape[3] - 1) + kernel_size[1]
            )

        # Apply the padding (crop the output)
        if padding[0] > 0:
            output = output[:, :, padding[0]:-padding[0], :]
        if padding[1] > 0:
            output = output[:, :, :, padding[1]:-padding[1]]

        # Store the input and the output for the backward propagation
        self.input = input
        self.output = output

        return output

    def backward(self, grad):
        # Check the input shape
        assert grad.shape == self.output.shape

        # Get the settings
        kernel_size = self.settings['kernel_size']
        stride = self.settings['stride']
        padding = self.settings['padding']
        dilation = self.settings['dilation']
        N, _, H, W, = self.input.shape

        # Reshape the weight
        weight_ = self.weight.transpose(0, 1).flip(dims=(2, 3,)).reshape(self.out_channels, -1)

        # Calculate the gradient wrt the input
        grad_ = self.pad(grad, padding=padding).reshape(N, self.out_channels, -1)
        gradx_ = weight_.transpose(0, 1) @ grad_
        gradx = fold(
            gradx_,
            output_size=(H + (H-1) * (stride[0] - 1), W + (W-1) * (stride[1] - 1)),
            kernel_size=kernel_size,
            padding=(kernel_size[0] - 1, kernel_size[1] - 1)
            )
        gradx = self.undilate(gradx, dilation=stride)
        assert gradx.shape == self.input.shape

        # Calculate and accumulate the gradients wrt the parameters
        input_ = unfold(
            self.dilate(self.input, dilation=stride),
            kernel_size=kernel_size,
            padding=(kernel_size[0] - 1, kernel_size[1] - 1)
            )
        gradw_ = (grad_ @ input_.transpose(1, 2)).sum(dim=0)
        self.gradw += gradw_.reshape(self.out_channels, self.in_channels, *kernel_size).transpose(0, 1)
        self.gradb += grad_.sum(dim=(0, 2))

        return gradx

    def dilate(self, a, dilation):
        # Add zero rows and columns between the elements
        N, C, H, W = a.shape
        b = empty(
            N, C, H + (H-1) * (dilation[0] - 1), W + (W-1) * (dilation[1] - 1), device=a.device).zero_()
        b[:, :, ::dilation[0], ::dilation[1]] = a
        return b

    def undilate(self, g, dilation):
        # Remove rows and columns between the elements
        h = g[:, :, ::dilation[0], ::dilation[1]]
        return h

    def pad(self, g, padding):
        # Add zero padding
        N, C, H, W = g.shape
        h = empty(
            N, C, H + 2 * padding[0], W + 2 * padding[1], device=g.device).zero_()
        h[:, :, padding[0]:H+padding[0], padding[1]:W+padding[1]] = g
        return h

    def param(self):
        return [
            (self.weight, self.gradw),
            (self.bias, self.gradb),
            ]

class SGD():
    def __init__(self, parameters, lr, weight_decay=0):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        # Apply a step of stochastic gradient decent on the parameters of the model
        for weight, grad in self.parameters:
            if self.weight_decay:
                # Calculate and apply weight decay
                grad = grad + self.weight_decay * weight
            weight.sub_(self.lr * grad)

    def zero_grad(self):
        # Set the gradients of the parameters to zero
        for _, gradp in self.parameters:
            gradp.zero_()
