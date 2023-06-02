"""
Implementation of a 3D W-Net model, based on the 2D version from https://arxiv.org/abs/1711.08506.
The model performs unsupervised segmentation of 3D images.
"""

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


class WNet_encoder(nn.Module):
    """WNet with encoder only."""

    def __init__(self, device, in_channels=1, out_channels=1, num_classes=2):
        super().__init__()
        self.device = device
        self.encoder = UNet(device, in_channels, num_classes, encoder=True)

    def forward(self, x):
        """Forward pass of the W-Net model."""
        return self.encoder(x)


class WNet(nn.Module):
    """Implementation of a 3D W-Net model, based on the 2D version from https://arxiv.org/abs/1711.08506.
    The model performs unsupervised segmentation of 3D images.
    It first encodes the input image into a latent space using the U-Net UEncoder, then decodes it back to the original image using the U-Net UDecoder.
    """

    def __init__(self, device, in_channels=1, out_channels=1, num_classes=2):
        super(WNet, self).__init__()
        self.device = device
        self.encoder = UNet(device, in_channels, num_classes, encoder=True)
        self.decoder = UNet(device, num_classes, out_channels, encoder=False)

    def forward(self, x):
        """Forward pass of the W-Net model."""
        enc = self.forward_encoder(x)
        dec = self.forward_decoder(enc)
        return enc, dec

    def forward_encoder(self, x):
        """Forward pass of the encoder part of the W-Net model."""
        return self.encoder(x)

    def forward_decoder(self, enc):
        """Forward pass of the decoder part of the W-Net model."""
        return self.decoder(enc)


class UNet(nn.Module):
    """Half of the W-Net model, based on the U-Net architecture."""

    def __init__(
        self, device, in_channels, out_channels, encoder=True, dropout=0.65
    ):
        super(UNet, self).__init__()
        self.device = device
        self.max_pool = nn.MaxPool3d(2)
        self.in_b = InBlock(device, in_channels, 64, dropout=dropout)
        self.conv1 = Block(device, 64, 128, dropout=dropout)
        self.conv2 = Block(device, 128, 256, dropout=dropout)
        self.conv3 = Block(device, 256, 512, dropout=dropout)
        self.bot = Block(device, 512, 1024, dropout=dropout)
        self.deconv1 = Block(device, 1024, 512, dropout=dropout)
        self.conv_trans1 = nn.ConvTranspose3d(
            1024, 512, 2, stride=2, device=self.device
        )
        self.deconv2 = Block(device, 512, 256, dropout=dropout)
        self.conv_trans2 = nn.ConvTranspose3d(
            512, 256, 2, stride=2, device=self.device
        )
        self.deconv3 = Block(device, 256, 128, dropout=dropout)
        self.conv_trans3 = nn.ConvTranspose3d(
            256, 128, 2, stride=2, device=self.device
        )
        self.out_b = OutBlock(device, 128, out_channels, dropout=dropout)
        self.conv_trans_out = nn.ConvTranspose3d(
            128, 64, 2, stride=2, device=self.device
        )

        self.sm = nn.Softmax(dim=1).to(device)
        self.encoder = encoder

    def forward(self, x):
        """Forward pass of the U-Net model."""
        in_b = self.in_b(x.to(self.device))
        c1 = self.conv1(self.max_pool(in_b))
        c2 = self.conv2(self.max_pool(c1))
        c3 = self.conv3(self.max_pool(c2))
        x = self.bot(self.max_pool(c3))
        x = self.deconv1(
            torch.cat(
                [
                    c3,
                    self.conv_trans1(x),
                ],
                dim=1,
            )
        )
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
        if self.encoder:
            x = self.sm(x)
        return x


class InBlock(nn.Module):
    """Input block of the U-Net architecture."""

    def __init__(self, device, in_channels, out_channels, dropout=0.65):
        super(InBlock, self).__init__()
        self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm3d(out_channels, device=device),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm3d(out_channels, device=device),
        ).to(device)

    def forward(self, x):
        """Forward pass of the input block."""
        return self.module(x.to(self.device))


class Block(nn.Module):
    """Basic block of the U-Net architecture."""

    def __init__(self, device, in_channels, out_channels, dropout=0.65):
        super(Block, self).__init__()
        self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, device=device),
            nn.Conv3d(in_channels, out_channels, 1, device=device),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm3d(out_channels, device=device),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, device=device),
            nn.Conv3d(out_channels, out_channels, 1, device=device),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm3d(out_channels, device=device),
        ).to(device)

    def forward(self, x):
        """Forward pass of the basic block."""
        return self.module(x.to(self.device))


class OutBlock(nn.Module):
    """Output block of the U-Net architecture."""

    def __init__(self, device, in_channels, out_channels, dropout=0.65):
        super(OutBlock, self).__init__()
        self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm3d(64, device=device),
            nn.Conv3d(64, 64, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm3d(64, device=device),
            nn.Conv3d(64, out_channels, 1, device=device),
        ).to(device)

    def forward(self, x):
        """Forward pass of the output block."""
        return self.module(x.to(self.device))
