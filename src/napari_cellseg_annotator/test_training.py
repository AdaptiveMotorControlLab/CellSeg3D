import torch
from torch import nn
import torcheck
import skimage.io as skio
from napari_cellseg_annotator import utils
import numpy as np
import napari
from napari_cellseg_annotator.model_3D_UNET_TRAILMAP import Unet_3d, train


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
#     device,
#     epochs=4,
# )

tl = train(model, train_loader, loss_fn, optimizer,device, epochs=5)

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