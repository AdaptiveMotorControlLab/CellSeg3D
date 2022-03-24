from matplotlib.style import use
import torch
from torch import nn
from torch.utils.data import DataLoader
import skimage.io as skio
from napari_cellseg_annotator import utils
import numpy as np
import napari




# Get cpu or gpu device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

print(torch.__version__)

##############
# for testing only, messes up with cuda
use_torcheck = False
if use_torcheck and device == "cpu" :
    import torcheck
##############

# device = torch.device("cuda")

# vol_dir= "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/volumes/images.tif"
# lab_dir = "C:/Users/Cyril/Desktop/Proj_bachelor/data/visual_tif/labels/testing_im.tif"


train_path_vol = "C:/Users/Cyril/Desktop/Proj_bachelor/code/pytorch-test-3dunet/cropped_visual/train/vol"
train_path_lab = "C:/Users/Cyril/Desktop/Proj_bachelor/code/pytorch-test-3dunet/cropped_visual/train/lab"

train_volumes = utils.load_images_unet(train_path_vol, ".tif")
train_labels = utils.load_images_unet(train_path_lab, ".tif")


val_path_vol = "C:/Users/Cyril/Desktop/Proj_bachelor/code/pytorch-test-3dunet/cropped_visual/val/vol/crop_vol_val.tif"
val_path_lab = "C:/Users/Cyril/Desktop/Proj_bachelor/code/pytorch-test-3dunet/cropped_visual/val/lab/crop_lab_val.tif"


test_label = np.asarray(skio.imread(val_path_lab))
test_vol = np.asarray(skio.imread(val_path_vol))

X_train = torch.stack(
    [torch.from_numpy(np.array(i.astype(np.uint8))) for i in train_volumes]
).to(device)
y_train = torch.stack(
    [torch.from_numpy(np.array(i.astype(np.uint8))) for i in train_labels]
).to(device)
X_val = torch.stack([torch.from_numpy(np.array(i.astype(np.uint8))) for i in test_vol])
y_val = torch.stack(
    [torch.from_numpy(np.array(i.astype(np.uint8))) for i in test_label]
).to(device)

# reshape into [C, H, W]
print(X_train.shape)

X_train = X_train.reshape((-1, 1, 64, 64, 64)).float().to(device)
y_train = y_train.reshape((-1, 1, 64, 64, 64)).float().to(device)
X_val = X_val.reshape((-1, 1, 64, 64, 64)).float().to(device)
y_val = y_val.reshape((-1, 1, 64, 64, 64)).float().to(device)
print(X_train.shape)

# create dataset and dataloaders

BATCH_SIZE = 1

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)




class Unet_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        return out[0].to(device)

    def encoderBlock(self, in_ch, out_ch, kernel_size, padding="same"):

        encode = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
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
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
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
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
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


model = Unet_3d(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

###################################
# torcheck code
#DISABLE WITH CUDA (uses CPU)
if use_torcheck and device == "cpu" :
    torcheck.register(optimizer)
    torcheck.add_module_changing_check(model, module_name="3D UNET test")
    torcheck.add_module_output_range_check(model, output_range=(0, 1), negate_range=True)
    torcheck.add_module_nan_check(model)
    torcheck.add_module_inf_check(model)
###################################


def train(model, train_dl, loss_fn, optimizer, epochs=2):

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


loss_fn = nn.CrossEntropyLoss()
xs = 64
ys = 64
x1 = torch.rand(1, 1, 1, xs, xs, xs)  # batch size, channel, d, h, w
y1 = torch.rand([1, 1, ys, ys, ys])
x1 = x1.float().to(device)
y1 = y1.float().to(device)

train_dataset_rand = torch.utils.data.TensorDataset(x1, y1)
# tl = train(
#     model,
#     train_dataset_rand,
#     loss_fn,
#     optimizer,
#     epochs=4,
# )

tl = train(model, train_loader, loss_fn, optimizer, epochs=5)

t = torch.rand(1, 1, 64, 64, 64).to(device)
# t_plot = t.detach().numpy()

# plt.imshow( t_plot[0][0][0])
# plt.show()

# input_im = t.to(device)
input_im = X_val

model.eval()
res = model(input_im)
# print(res)
res = res.cpu().detach().numpy()
print(res.shape)
# print(res[0][0])

print("=============================")
print("Done !")
viewer = napari.Viewer(ndisplay=3)
multiscale_layer = viewer.add_image(res[0], colormap="red",name="pred", scale=[1, 1, 1])
original_layer = viewer.add_image(
    test_vol, colormap="bop blue",name="original", scale=[1, 1, 1], opacity=0.7
)
label_layer = viewer.add_labels(
    test_label, name="original labels",
)
napari.run()
