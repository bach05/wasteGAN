import torch.utils.data
from torch.nn import functional as F
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    """
    ## Discriminator Loss
    We want to find $w$ to maximize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +
     \frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        """
        * `f_real` is $f_w(x)$
        * `f_fake` is $f_w(g_\theta(z))$
        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
        which are later added.
        They are kept separate for logging.
        """

        # We use ReLUs to clip the loss to keep $f \in [-1, +1]$ range.
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class DiscriminatorLoss2(nn.Module):
    """
    ## Discriminator Loss
    We want to find $w$ to maximize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +
     \frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor, offset=0.5):
        """
        * `f_real` is $f_w(x)$
        * `f_fake` is $f_w(g_\theta(z))$
        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
        which are later added.
        They are kept separate for logging.
        """

        # We use ReLUs to clip the loss to keep $f \in [-1, +1]$ range.
        return F.relu(offset - f_real).mean(), F.relu(offset + f_fake).mean()

class SoftDiscriminatorLoss(nn.Module):
    """
    ## Discriminator Loss
    """

    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        """
        """
        real_loss = F.softplus(-f_real)
        fake_loss = F.softplus(f_fake)

        return real_loss.mean(), fake_loss.mean()


class GeneratorLoss(nn.Module):
    """
    ## Generator Loss
    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_fake: torch.Tensor):
        """
        * `f_fake` is $f_w(g_\theta(z))$
        """
        return -f_fake.mean()

class GeneratorLoss2(nn.Module):
    """
    ## Generator Loss
    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_fake: torch.Tensor):
        """
        * `f_fake` is $f_w(g_\theta(z))$
        """
        return (-f_fake+1).mean()

class GeneratorLoss3(nn.Module):
    """
    ## Generator Loss
    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor, w_real: float = 0.2, w_fake: float = 0.8):
        """
        * `f_fake` is $f_w(g_\theta(z))$
        """
        return (-w_fake * f_fake + w_real * f_real).mean()