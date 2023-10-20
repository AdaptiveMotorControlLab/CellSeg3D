"""Legacy version of adapted TRAILMAP model, not used in the current version of the plugin."""
# import torch
# from torch import nn
#
#
# class TRAILMAP(nn.Module):
#     def __init__(self, in_ch, out_ch, *args, **kwargs):
#         super().__init__()
#         self.conv0 = self.encoderBlock(in_ch, 32, 3)  # input
#         self.conv1 = self.encoderBlock(32, 64, 3)  # l1
#         self.conv2 = self.encoderBlock(64, 128, 3)  # l2
#         self.conv3 = self.encoderBlock(128, 256, 3)  # l3
#
#         self.bridge = self.bridgeBlock(256, 512, 3)
#
#         self.up5 = self.decoderBlock(256 + 512, 256, 2)
#
#         self.up6 = self.decoderBlock(128 + 256, 128, 2)
#         self.up7 = self.decoderBlock(128 + 64, 64, 2)  # l2
#         self.up8 = self.decoderBlock(64 + 32, 32, 2)  # l1
#         self.out = self.outBlock(32, out_ch, 1)
#
#     def forward(self, x):
#         conv0 = self.conv0(x)  # l0
#         conv1 = self.conv1(conv0)  # l1
#         conv2 = self.conv2(conv1)  # l2
#         conv3 = self.conv3(conv2)  # l3
#
#         bridge = self.bridge(conv3)  # bridge
#         # print("bridge :")
#         # print(bridge.shape)
#
#         up5 = self.up5(torch.cat([conv3, bridge], 1))  # l3
#         # print("up")
#         # print(up5.shape)
#         up6 = self.up6(torch.cat([up5, conv2], 1))  # l2
#         # print(up6.shape)
#         up7 = self.up7(torch.cat([up6, conv1], 1))  # l1
#         # print(up7.shape)
#
#         up8 = self.up8(torch.cat([up7, conv0], 1))  # l1
#         # print(up8.shape)
#         return self.out(up8)
#         # print("out:")
#         # print(out.shape)
#
#     def encoderBlock(self, in_ch, out_ch, kernel_size, padding="same"):
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#             nn.Conv3d(
#                 out_ch, out_ch, kernel_size=kernel_size, padding=padding
#             ),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#         )
#
#     def bridgeBlock(self, in_ch, out_ch, kernel_size, padding="same"):
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#             nn.Conv3d(
#                 out_ch, out_ch, kernel_size=kernel_size, padding=padding
#             ),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#         )
#
#     def decoderBlock(self, in_ch, out_ch, kernel_size, padding="same"):
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#             nn.Conv3d(
#                 out_ch, out_ch, kernel_size=kernel_size, padding=padding
#             ),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#             nn.ConvTranspose3d(
#                 out_ch, out_ch, kernel_size=kernel_size, stride=(2, 2, 2)
#             ),
#         )
#
#     def outBlock(self, in_ch, out_ch, kernel_size, padding="same"):
#         return nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#         )
#
#
# class TRAILMAP_(TRAILMAP):
#     weights_file = "TRAILMAP_PyTorch.pth"  # model additionally trained on Mathis/Wyss mesoSPIM data
#     # FIXME currently incorrect, find good weights from TRAILMAP_test and upload them
#
#     def __init__(self, in_channels=1, out_channels=1, **kwargs):
#         super().__init__(in_channels, out_channels, **kwargs)
#
#     # def get_output(model, input):
#     #     out = model(input)
#     #
#     #     return out
#
#     # def get_validation(model, val_inputs):
#     #     return model(val_inputs)
