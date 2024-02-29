import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8, stride_step: int = 1):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        self.stride_step = stride_step

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b, stride = self.stride_step)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        #return x.reshape(-1, self.out_features, h, w)
        return x


class StyleBlock(nn.Module):
    """
    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int, stride_step=2):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3, stride_step=stride_step)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tensor of shape `[batch_size, 1, height, width]`
        """
        # Get style vector $s$
        s = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, s)
        # Scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(3))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        #print(f"RGB: bias {self.bias.shape}, x {x.shape}")
        return self.activation(x + self.bias[None, :, None, None])


class ToSEG(nn.Module):
    def __init__(self, d_latent: int, features: int, n_classes: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, features//2, kernel_size=3, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(features//2))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

        #second conv
        self.conv2 = nn.Conv2d(features//2, n_classes, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        #print(f"SEG: bias {self.bias.shape}, x {x.shape}")
        x = self.activation(x + self.bias[None, :, None, None])

        return self.conv2(x)


class MaskGeneratorBlock(nn.Module):

    def __init__(self, d_latent: int, in_features: int, out_features: int, n_classes: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()

        # First [style block](#style_block) changes the feature map size to `out_features`
        #self.style_block1 = StyleBlock(d_latent, in_features, out_features//2)  #MOD
        self.style_block1 = StyleBlock(d_latent, in_features, out_features, stride_step=2)  #MOD
        # Second [style block](#style_block)
        #self.style_block2 = StyleBlock(d_latent, out_features//2, out_features)
        #self.style_block3 = StyleBlock(d_latent, out_features, out_features) #MOD
        self.style_block2 = StyleBlock(d_latent, out_features, out_features, stride_step=1) #MOD

        # *toRGB* layer
        #self.to_rgb = ToRGB(d_latent, out_features)
        #self.to_seg = ToSEG(d_latent, out_features, n_classes=n_classes)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tuple of two noise tensors of shape `[batch_size, 1, height, width]`
        """
        # First style block with first noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block1(x, w, noise[0])
        # Second style block with second noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block2(x, w, noise[1])
        #x = self.style_block3(x, w, noise[2]) #MOD

        # Get RGB image
        #rgb = self.to_rgb(x, w)
        #seg = self.to_seg(x, w)

        # Return feature map and rgb image
        return x

class FeatMaskGenerator(nn.Module):

    def __init__(self, features: List, d_latent: int, n_classes: int, input_res: int = 256):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `d_latent` is the dimensionality of $w$
        * `n_features` number of features in the convolution layer at the highest resolution (final block)
        * `input_res` resolution of the original image
        """
        super().__init__()

        print("**** GEN FEAT")

        # Number of generator blocks
        self.n_blocks = len(features)

        # Trainable $4 \times 4$ constant
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], input_res, input_res)))

        # First style block for $4 \times 4$ resolution and layer to get RGB
        self.style_block = StyleBlock(d_latent, features[0], features[0], stride_step=1)
        #self.to_rgb = ToRGB(d_latent, features[0])
        #self.to_seg = ToSEG(d_latent, features[0], n_classes=n_classes)

        # Generator blocks
        blocks = [MaskGeneratorBlock(d_latent, features[i - 1], features[i], n_classes=n_classes) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # $2 \times$ up sampling layer. The feature space is up sampled
        # at each block
        #self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        * `w` is $w$. In order to mix-styles (use different $w$ for different layers), we provide a separate
        $w$ for each [generator block](#generator_block). It has shape `[n_blocks, batch_size, d_latent]`.
        * `input_noise` is the noise for each block.
        It's a list of pairs of noise sensors because each block (except the initial) has two noise inputs
        after each convolution layer (see the diagram).
        """

        # Get batch size
        batch_size = w.shape[1]

        # Expand the learned constant to match batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # The first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # Get first rgb image
        #rgb = self.to_rgb(x, w[0])
        #seg = self.to_seg(x, w[0])

        # Evaluate rest of the blocks
        internal_states = []
        for i in range(1, self.n_blocks):
            # Up sample the feature map
            x# = self.up_sample(x)
            # Run it through the [generator block](#generator_block)
            x = self.blocks[i - 1](x, w[i], input_noise[i])
            # Up sample the RGB image and add to the rgb from the block
            #rgb = self.up_sample(rgb) + rgb_new
            #seg = self.up_sample(seg) + seg_new
            internal_states.append(x)

        # Return the final RGB image and the seg mask
        return x, internal_states
    






###############################################
# COMMON
###############################################


class EqualizedLinear(nn.Module):
    """
    This uses [learning-rate equalized weights](#equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    This uses [learning-rate equalized weights](#equalized_weights) for a convolution layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """
    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\mathcal{N}(0,c)$ they initialize weights
    to $\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \hat{w}_i$$

    The gradients on stored parameters $\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.

    The optimizer updates on $\hat{w}$ are proportionate to the learning rate $\lambda$.
    But the effective weights $w$ get updated proportionately to $c \lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\lambda$.

    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """

    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c


class GradientPenalty(nn.Module):
    """
    This is the $R_1$ regularization penality from the paper
    [Which Training Methods for GANs do actually Converge?](https://papers.labml.ai/paper/1801.04406).

    $$R_1(\psi) = \frac{\gamma}{2} \mathbb{E}_{p_\mathcal{D}(x)}
    \Big[\Vert \nabla_x D_\psi(x)^2 \Vert\Big]$$

    That is we try to reduce the L2 norm of gradients of the discriminator with
    respect to images, for real images ($P_\mathcal{D}$).
    """

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        * `x` is $x \sim \mathcal{D}$
        * `d` is $D(x)$
        """

        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of $D(x)$ with respect to $x$.
        # `grad_outputs` is set to $1$ since we want the gradients of $D(x)$,
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        # Calculate the norm $\Vert \nabla_{x} D(x)^2 \Vert$
        norm = gradients.norm(2, dim=-1)
        # Return the loss $\Vert \nabla_x D_\psi(x)^2 \Vert$
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    """
    This regularization encourages a fixed-size step in $w$ to result in a fixed-magnitude
    change in the image.

    $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
      \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$

    where $\mathbf{J}_w$ is the Jacobian
    $\mathbf{J}_w = \frac{\partial g}{\partial w}$,
    $w$ are sampled from $w \in \mathcal{W}$ from the mapping network, and
    $y$ are images with noise $\mathcal{N}(0, \mathbf{I})$.

    $a$ is the exponential moving average of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
    as the training progresses.

    $\mathbf{J}^\top_{w} y$ is calculated without explicitly calculating the Jacobian using
    $$\mathbf{J}^\top_{w} y = \nabla_w \big(g(w) \cdot y \big)$$
    """

    def __init__(self, beta: float):
        """
        * `beta` is the constant $\beta$ used to calculate the exponential moving average $a$
        """
        super().__init__()

        # $\beta$
        self.beta = beta
        # Number of steps calculated $N$
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # Exponential sum of $\mathbf{J}^\top_{w} y$
        # $$\sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
        # where $[\mathbf{J}^\top_{w} y]_i$ is the value of it at $i$-th step of training
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        * `w` is the batch of $w$ of shape `[batch_size, d_latent]`
        * `x` are the generated images of shape `[batch_size, 3, height, width]`
        """

        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        # Calculate $y \in \mathcal{N}(0, \mathbf{I})$
        y = torch.randn(x.shape, device=device)
        # Calculate $\big(g(w) \cdot y \big)$ and normalize by the square root of image size.
        # This is scaling is not mentioned in the paper but was present in
        # [their implementation](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167).
        output = (x * y).sum() / math.sqrt(image_size)

        # Calculate gradients to get $\mathbf{J}^\top_{w} y$
        #print("output grad: ", output.grad)
        #print("w grad: ", w.grad)
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        # Calculate L2-norm of $\mathbf{J}^\top_{w} y$
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            # Calculate $a$
            # $$\frac{1}{1 - \beta^N} \sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            # Calculate the penalty
            # $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
            # \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        # Calculate the mean of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment $N$
        self.steps.add_(1.)

        #print("plp: ",loss)

        # Return the penalty
        return loss

######################
# NO styleGAN
######################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, k_size=3, pad=1):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=k_size, padding=pad, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=k_size, padding=pad, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            #print("x res: ", x.shape)
            #print("f(x) res: ", self.double_conv(x).shape)
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)