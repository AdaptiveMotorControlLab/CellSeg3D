import torch
from torch import nn

BATCH_SIZE = 1

class Unet_3d(nn.Module):
    def __init__(self, in_ch, out_ch, device):
        super().__init__()
        self.device = device
        self.conv0 = self.encoderBlock(in_ch, 32, 3)  # input
        self.conv1 = self.encoderBlock(32, 64, 3)  # l1
        self.conv2 = self.encoderBlock(64, 128, 3)  # l2
        self.conv3 = self.encoderBlock(128, 256, 3)  # l3
        # self.conv4 = self.encoderBlock(256, 512, 3)

        self.bridge = self.bridgeBlock(256, 512, 3)
        # self.bridge = self.bridgeBlock(512, 1024, 3)

        # self.up5 = self.decoderBlock(512 + 1024, 512, 2)  # l4
        self.up5 = self.decoderBlock(256 + 512, 256, 2)
        # self.up6 = self.decoderBlock(512 + 256, 256, 2)  # l3
        self.up6 = self.decoderBlock(128 + 256, 128, 2)
        # self.up7 = self.decoderBlock(128 + 256, 128, 2)
        self.up7 = self.decoderBlock(128 + 64, 64, 2)  # l2
        self.up8 = self.decoderBlock(64 + 32, 32, 2)  # l1
        self.out = self.outBlock(32, out_ch, 1)

    def forward(self, x):
        conv0 = self.conv0(x)  # l0
        conv1 = self.conv1(conv0)  # l1
        conv2 = self.conv2(conv1)  # l2
        conv3 = self.conv3(conv2)  # l3
        # conv4 = self.conv4(conv3)
        print("x")
        print(x.shape)
        print("down")
        print(conv1.shape)
        print(conv2.shape)
        print(conv3.shape)
        # print(conv4.shape)

        bridge = self.bridge(conv3)  # bridge
        print("bridge :")
        print(bridge.shape)

        up5 = self.up5(torch.cat([conv3, bridge], 1))  # l3
        print("up")
        print(up5.shape)
        up6 = self.up6(torch.cat([up5, conv2], 1))  # l2
        print(up6.shape)
        up7 = self.up7(torch.cat([up6, conv1], 1))  # l1
        print(up7.shape)

        up8 = self.up8(torch.cat([up7, conv0], 1))  # l1
        print(up8.shape)
        out = self.out(up8)
        print("out:")
        print(out.shape)
        return out[0].to(self.device)

    def encoderBlock(self, in_ch, out_ch, kernel_size, padding="same"):

        encode = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(
                out_ch, out_ch, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        return encode

    def bridgeBlock(self, in_ch, out_ch, kernel_size, padding="same"):

        encode = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(
                out_ch, out_ch, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            # nn.ConvTranspose3d(out_ch, out_ch, kernel_size=2, stride=2),
        )
        return encode

    def decoderBlock(self, in_ch, out_ch, kernel_size, padding="same"):

        decode = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(
                out_ch, out_ch, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.ConvTranspose3d(
                out_ch, out_ch, kernel_size=kernel_size, stride=(2, 2, 2)
            ),
        )
        return decode

    def outBlock(self, in_ch, out_ch, kernel_size, padding="same"):

        out = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm3d(out_ch),
        )
        return out


def train(model, train_dl, loss_fn, optimizer, device, epochs=2):

    model.to(device)

    train_loss = []

    best_acc = 0.0

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        model.train(True)  # Set training mode = true
        dataloader = train_dl

        running_loss = 0.0
        running_acc = 0.0

        step = 0

        # iterate over data
        # print("loader :")
        # print(dataloader.size())
        for x, y in dataloader:
            # print("test :")
            # print(x.size())
            # print(y.size())
            # print(f'x = min: {x.min()}; max: {x.max()}')
            # print(f'y = min: {y.min()}; max: {y.max()}')
            x = x.to(device)
            y = y.to(device)
            step += 1

            # forward pass

            # zero the gradients

            outputs = model(x)
            print("out and y:")
            # print(outputs.shape)
            # print(outputs.type())
            # print(outputs.device)
            # print(y.shape)
            # print(y.type())
            # print(y.device)

            loss = loss_fn(outputs, y[0])
            # print("loss :")
            # print(loss)

            # the backward pass frees the graph memory, so there is no
            # need for torch.no_grad in this training pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # stats - whatever is the phase
            # acc = acc_fn(outputs, y)
            batch_size = BATCH_SIZE
            # running_acc += acc * batch_size
            running_loss += loss * batch_size

            # if step % 10 == 0:
            #     # clear_output(wait=True)
            #     print(
            #         "Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}".format(
            #             step,
            #             loss,
            #             acc,
            #             torch.device.memory_allocated() / 1024 / 1024,
            #         )
            #     )
            # print(torch.cuda.memory_summary())

        epoch_loss = running_loss
        print("--------------------")
        print(f"Current loss: {loss}")
        print("--------------------")
        # epoch_acc = running_acc

        train_loss.append(epoch_loss)

    return train_loss
