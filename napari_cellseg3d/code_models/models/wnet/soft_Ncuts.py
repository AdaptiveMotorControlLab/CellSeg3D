"""
Implementation of a 3D Soft N-Cuts loss based on https://arxiv.org/abs/1711.08506 and https://ieeexplore.ieee.org/document/868688.
The implementation was adapted and approximated to reduce computational and memory cost.
This faster version was proposed on https://github.com/fkodom/wnet-unsupervised-image-segmentation.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

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
        print(f"Radius set to {self.radius}")

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

    # def get_distances(self):
    #     """Precompute the spatial distance of the pixels for the weights calculation, to avoid recomputing it at each iteration.
    #
    #     Returns:
    #         distances (dict): for each pixel index, we get the distances to the pixels in a radius around it.
    #     """
    #
    #     distances = dict()
    #     indexes = np.array(
    #         [
    #             (i, j, k)
    #             for i in range(self.H)
    #             for j in range(self.W)
    #             for k in range(self.D)
    #         ]
    #     )
    #
    #     for i in indexes:
    #         iTuple = (i[0], i[1], i[2])
    #         distances[iTuple] = dict()
    #
    #         sliceD = indexes[
    #             i[0] * self.H
    #             + i[1] * self.W
    #             + max(0, i[2] - self.radius) : i[0] * self.H
    #             + i[1] * self.W
    #             + min(self.D, i[2] + self.radius)
    #         ]
    #         sliceW = indexes[
    #             i[0] * self.H
    #             + max(0, i[1] - self.radius) * self.W
    #             + i[2] : i[0] * self.H
    #             + min(self.W, i[1] + self.radius) * self.W
    #             + i[2] : self.D
    #         ]
    #         sliceH = indexes[
    #             max(0, i[0] - self.radius) * self.H
    #             + i[1] * self.W
    #             + i[2] : min(self.H, i[0] + self.radius) * self.H
    #             + i[1] * self.W
    #             + i[2] : self.D * self.W
    #         ]
    #
    #         for j in np.concatenate((sliceD, sliceW, sliceH)):
    #             jTuple = (j[0], j[1], j[2])
    #             distance = np.linalg.norm(i - j)
    #             if distance > self.radius:
    #                 continue
    #             distance = math.exp(
    #                 -(distance**2) / (self.spatial_sigma**2)
    #             )
    #
    #             if jTuple not in distances:
    #                 distances[iTuple][jTuple] = distance
    #
    #     return distances, indexes

    # def get_weights(self, inputs):
    #     """Computes the weights matrix for the Soft N-Cuts loss.
    #
    #     Args:
    #         inputs (torch.Tensor): Tensor of shape (N, C, H, W, D) containing the input images.
    #
    #     Returns:
    #         list: List of the weights dict for each image in the batch.
    #     """
    #
    #     # Compute the brightness distance of the pixels
    #     flatted_inputs = inputs.view(
    #         inputs.shape[0], inputs.shape[1], -1
    #     )  # (N, C, H*W*D)
    #     I_diff = torch.subtract(
    #         flatted_inputs.unsqueeze(3), flatted_inputs.unsqueeze(2)
    #     )  # (N, C, H*W*D, H*W*D)
    #     masked_I_diff = torch.mul(I_diff, self.mask)  # (N, C, H*W*D, H*W*D)
    #     squared_I_diff = torch.pow(masked_I_diff, 2)  # (N, C, H*W*D, H*W*D)
    #
    #     W_I = torch.exp(
    #         torch.neg(torch.div(squared_I_diff, self.intensity_sigma))
    #     )  # (N, C, H*W*D, H*W*D)
    #     W_I = torch.mul(W_I, self.mask)  # (N, C, H*W*D, H*W*D)
    #
    #     # Get the spatial distance of the pixels
    #     unsqueezed_W_X = self.W_X.view(
    #         1, 1, self.W_X.shape[0], self.W_X.shape[1]
    #     )  # (1, 1, H*W*D, H*W*D)
    #
    #     return torch.mul(W_I, unsqueezed_W_X)  # (N, C, H*W*D, H*W*D)
