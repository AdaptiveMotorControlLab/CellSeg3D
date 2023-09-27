"""Implementation of a 3D Soft N-Cuts loss based on https://arxiv.org/abs/1711.08506 and https://ieeexplore.ieee.org/document/868688.

The implementation was adapted and approximated to reduce computational and memory cost.
This faster version was proposed on https://github.com/fkodom/wnet-unsupervised-image-segmentation.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

from napari_cellseg3d.utils import LOGGER as logger

__author__ = "Yves Paychère, Colin Hofmann, Cyril Achard"
__credits__ = [
    "Yves Paychère",
    "Colin Hofmann",
    "Cyril Achard",
    "Xide Xia",
    "Brian Kulis",
    "Jianbo Shi",
    "Jitendra Malik",
    "Frank Odom",
]


class SoftNCutsLoss(nn.Module):
    """Implementation of a 3D Soft N-Cuts loss based on https://arxiv.org/abs/1711.08506 and https://ieeexplore.ieee.org/document/868688.

    Args:
        data_shape (H, W, D): shape of the images as a tuple.
        intensity_sigma (scalar): scale of the gaussian kernel of pixels brightness.
        spatial_sigma (scalar): scale of the gaussian kernel of pixels spacial distance.
        radius (scalar): radius of pixels for which we compute the weights
    """

    def __init__(
        self, data_shape, device, intensity_sigma, spatial_sigma, radius=None
    ):
        """Initialize the Soft N-Cuts loss.

        Args:
            data_shape (H, W, D): shape of the images as a tuple.
            device (torch.device): device on which the loss is computed.
            intensity_sigma (scalar): scale of the gaussian kernel of pixels brightness.
            spatial_sigma (scalar): scale of the gaussian kernel of pixels spacial distance.
            radius (scalar): radius of pixels for which we compute the weights
        """
        super(SoftNCutsLoss, self).__init__()
        self.intensity_sigma = intensity_sigma
        self.spatial_sigma = spatial_sigma
        self.radius = radius
        self.H = data_shape[0]
        self.W = data_shape[1]
        self.D = data_shape[2]
        self.device = device

        if self.radius is None:
            self.radius = min(
                max(5, math.ceil(min(self.H, self.W, self.D) / 20)),
                self.H,
                self.W,
                self.D,
            )
        logger.info(f"Radius set to {self.radius}")

    def forward(self, labels, inputs):
        """Forward pass of the Soft N-Cuts loss.

        Args:
            labels (torch.Tensor): Tensor of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
            inputs (torch.Tensor): Tensor of shape (N, C, H, W, D) containing the input images.

        Returns:
            The Soft N-Cuts loss of shape (N,).
        """
        # inputs.shape[0]
        # inputs.shape[1]
        K = labels.shape[1]

        labels.to(self.device)
        inputs.to(self.device)

        loss = 0

        kernel = self.gaussian_kernel(self.radius, self.spatial_sigma).to(
            self.device
        )

        for k in range(K):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(
                inputs * class_probs, dim=(2, 3, 4), keepdim=True
            ) / torch.add(
                torch.mean(class_probs, dim=(2, 3, 4), keepdim=True), 1e-5
            )
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(
                diff.pow(2).mul(-1 / self.intensity_sigma**2)
            )

            numerator = torch.sum(
                class_probs
                * F.conv3d(class_probs * weights, kernel, padding=self.radius),
                dim=(1, 2, 3, 4),
            )
            denominator = torch.sum(
                class_probs * F.conv3d(weights, kernel, padding=self.radius),
                dim=(1, 2, 3, 4),
            )
            loss += nn.L1Loss()(
                numerator / torch.add(denominator, 1e-6),
                torch.zeros_like(numerator),
            )

        return K - loss

    def gaussian_kernel(self, radius, sigma):
        """Computes the Gaussian kernel.

        Args:
            radius (int): The radius of the kernel.
            sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
            The Gaussian kernel of shape (1, 1, 2*radius+1, 2*radius+1, 2*radius+1).
        """
        x_2 = np.linspace(-radius, radius, 2 * radius + 1) ** 2
        dist = (
            np.sqrt(
                x_2.reshape(-1, 1, 1)
                + x_2.reshape(1, -1, 1)
                + x_2.reshape(1, 1, -1)
            )
            / sigma
        )
        kernel = norm.pdf(dist) / norm.pdf(0)
        kernel = torch.from_numpy(kernel.astype(np.float32))
        return kernel.view(
            (1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2])
        )
