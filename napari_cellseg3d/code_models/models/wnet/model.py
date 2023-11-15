"""Implementation of a 3D W-Net model, based on the 2D version from https://arxiv.org/abs/1711.08506. The model performs unsupervised segmentation of 3D images."""

from typing import List

import torch
import torch.nn as nn

__author__ = "Yves Paychère, Colin Hofmann, Cyril Achard"
__credits__ = [
    "Yves Paychère",
    "Colin Hofmann",
    "Cyril Achard",
    "Xide Xia",
    "Brian Kulis",
]
NUM_GROUPS = 4


class WNet_encoder(nn.Module):
    """WNet with encoder only."""

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        # num_classes=2,
        softmax=True,
    ):
        """Initialize the W-Net encoder."""
        super().__init__()
        self.encoder = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            softmax=softmax,
        )

    def forward(self, x):
        """Forward pass of the W-Net model."""
        return self.encoder(x)


class WNet(nn.Module):
    """Implementation of a 3D W-Net model, based on the 2D version from https://arxiv.org/abs/1711.08506.

    The model performs unsupervised segmentation of 3D images.
    It first encodes the input image into a latent space using the U-Net UEncoder, then decodes it back to the original image using the U-Net UDecoder.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        num_classes=2,
        dropout=0.65,
    ):
        """Initialize the W-Net model."""
        super(WNet, self).__init__()
        self.encoder = UNet(
            in_channels, num_classes, softmax=True, dropout=dropout
        )
        self.decoder = UNet(
            num_classes, out_channels, softmax=False, dropout=dropout
        )

    def forward(self, x):
        """Forward pass of the W-Net model. Returns the segmentation and the reconstructed image."""
        enc = self.forward_encoder(x)
        return enc, self.forward_decoder(enc)

    def forward_encoder(self, x):
        """Forward pass of the encoder part of the W-Net model."""
        return self.encoder(x)

    def forward_decoder(self, enc):
        """Forward pass of the decoder part of the W-Net model."""
        return self.decoder(enc)


class UNet(nn.Module):
    """Half of the W-Net model, based on the U-Net architecture."""

    def __init__(
        self,
        # device,
        in_channels: int,
        out_channels: int,
        channels: List[int] = None,
        softmax: bool = True,
        dropout: float = 0.65,
    ):
        """Creates a U-Net model, which is half of the W-Net model."""
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        if len(channels) != 5:
            raise ValueError(
                "Channels must be a list of channels in the form: [64, 128, 256, 512, 1024]"
            )
        super(UNet, self).__init__()
        # self.device = device
        self.channels = channels
        self.max_pool = nn.MaxPool3d(2)
        self.in_b = InBlock(in_channels, self.channels[0], dropout=dropout)
        self.conv1 = Block(channels[0], self.channels[1], dropout=dropout)
        self.conv2 = Block(channels[1], self.channels[2], dropout=dropout)
        # self.conv3 = Block(channels[2], self.channels[3], dropout=dropout)
        # self.bot = Block(channels[3], self.channels[4], dropout=dropout)
        self.bot = Block(channels[2], self.channels[3], dropout=dropout)
        # self.bot = Block(channels[1], self.channels[2], dropout=dropout)
        # self.bot = Block(channels[0], self.channels[1], dropout=dropout)
        # self.deconv1 = Block(channels[4], self.channels[3], dropout=dropout)
        self.deconv2 = Block(channels[3], self.channels[2], dropout=dropout)
        self.deconv3 = Block(channels[2], self.channels[1], dropout=dropout)
        self.out_b = OutBlock(channels[1], out_channels, dropout=dropout)
        # self.conv_trans1 = nn.ConvTranspose3d(
        #     self.channels[4], self.channels[3], 2, stride=2
        # )
        self.conv_trans2 = nn.ConvTranspose3d(
            self.channels[3], self.channels[2], 2, stride=2
        )
        self.conv_trans3 = nn.ConvTranspose3d(
            self.channels[2], self.channels[1], 2, stride=2
        )
        self.conv_trans_out = nn.ConvTranspose3d(
            self.channels[1], self.channels[0], 2, stride=2
        )

        self.sm = nn.Softmax(dim=1)
        self.softmax = softmax

    def forward(self, x):
        """Forward pass of the U-Net model."""
        in_b = self.in_b(x)
        c1 = self.conv1(self.max_pool(in_b))
        c2 = self.conv2(self.max_pool(c1))
        # c3 = self.conv3(self.max_pool(c2))
        # x = self.bot(self.max_pool(c3))
        x = self.bot(self.max_pool(c2))
        # x = self.bot(self.max_pool(c1))
        # x = self.bot(self.max_pool(in_b))
        # x = self.deconv1(
        #     torch.cat(
        #         [
        #             c3,
        #             self.conv_trans1(x),
        #         ],
        #         dim=1,
        #     )
        # )
        x = self.deconv2(
            torch.cat(
                [
                    c2,
                    self.conv_trans2(x),
                ],
                dim=1,
            )
        )
        x = self.deconv3(
            torch.cat(
                [
                    c1,
                    self.conv_trans3(x),
                ],
                dim=1,
            )
        )
        x = self.out_b(
            torch.cat(
                [
                    in_b,
                    self.conv_trans_out(x),
                ],
                dim=1,
            )
        )
        if self.softmax:
            x = self.sm(x)
        return x


class InBlock(nn.Module):
    """Input block of the U-Net architecture."""

    def __init__(self, in_channels, out_channels, dropout=0.65):
        """Create the input block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float, optional): Dropout probability. Defaults to 0.65.
        """
        super(InBlock, self).__init__()
        # self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
        )

    def forward(self, x):
        """Forward pass of the input block."""
        return self.module(x)


class Block(nn.Module):
    """Basic block of the U-Net architecture."""

    def __init__(self, in_channels, out_channels, dropout=0.65):
        """Initialize the basic block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float, optional): Dropout probability. Defaults to 0.65.
        """
        super(Block, self).__init__()
        # self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.Conv3d(out_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
        )

    def forward(self, x):
        """Forward pass of the basic block."""
        return self.module(x)


class OutBlock(nn.Module):
    """Output block of the U-Net architecture."""

    def __init__(self, in_channels, out_channels, dropout=0.65):
        """Initialize the output block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float, optional): Dropout probability. Defaults to 0.65.
        """
        super(OutBlock, self).__init__()
        # self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(64),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=64),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(64),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=64),
            nn.Conv3d(64, out_channels, 1),
        )

    def forward(self, x):
        """Forward pass of the output block."""
        return self.module(x)
