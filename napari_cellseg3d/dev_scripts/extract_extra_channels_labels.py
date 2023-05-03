import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import expand_labels
from tqdm import tqdm


def extract_labels_from_channels(
    nucleus_labels: np.array,
    extra_channels: list,
    radius: int = 4,
    threshold_factor=2,
    viewer=None,
):
    """
    Attemps to extract labels from other channels by expanding nuclei labels and picking the one with most pixels around it.
    Args:
        nucleus_labels (np.array): labels for the nuclei
        extra_channels (list): channels arrays to extract labels from
        radius: radius in which the approximation is made

    Returns:
    A list of extracted labels for each extra channel
    """
    labeled_channels = {}

    contrasted_channels = []
    for channel in extra_channels:
        channel = (channel - np.min(channel)) / (
            np.max(channel) - np.min(channel)
        )
        threshold_brightness = threshold_otsu(channel) * threshold_factor
        channel_contrasted = np.where(
            channel > threshold_brightness, channel, 0
        )
        contrasted_channels.append(channel_contrasted)
        if viewer is not None:
            viewer.add_image(
                channel_contrasted,
                name="channel_contrasted",
                colormap="viridis",
            )
    for label_id in tqdm(np.unique(nucleus_labels)):
        if label_id == 0:
            continue
        label_nucleus = np.where(nucleus_labels == label_id, nucleus_labels, 0)
        expanded = expand_labels(label_nucleus, distance=radius)
        for i, channel in enumerate(contrasted_channels):
            label_contrasted = np.where(expanded != 0, channel, 0)
            labeled_channel = np.where(label_contrasted != 0, label_id, 0)
            labeled_channels[
                f"label_{label_id}_channel_{i+1}"
            ] = np.count_nonzero(labeled_channel)
            if np.count_nonzero(labeled_channel) > 0 and viewer is not None:
                print(np.count_nonzero(labeled_channel))
                viewer.add_labels(
                    labeled_channel, name=f"label_{label_id}_channel_{i+1}"
                )

    return labeled_channels


if __name__ == "__main__":
    from pathlib import Path

    import napari
    import pandas as pd
    from tifffile import imread

    image_path = (
        Path.home()
        / "Desktop/Code/WNet-benchmark/results/showcase/WNet-labels-Voronoi-Otsu.tif"
    )
    # image_path = Path.home() / "Desktop/Code/WNet-benchmark/results/showcase/WNet-labels-Voronoi-Otsu.tif"
    nuclei_path = (
        Path.home()
        / "Desktop/Code/WNet-benchmark/results/showcase/ELAST9_5_DAPI_Iba1_CD163_crop2_denoised__DAPI_only.tif"
    )
    extra_channels_path = (
        Path.home()
        / "Desktop/Code/WNet-benchmark/dataset/wyss_data/batch_1/tmp"
    )
    extra_channels = [
        imread(str(path))
        for path in extra_channels_path.glob(
            "ELAST9_5_DAPI_Iba1_CD163_crop2_denoised__*.tif"
        )
    ]
    labels = imread(str(image_path))
    viewer = napari.Viewer()

    shift = 0
    viewer.add_image(
        imread(str(nuclei_path))[
            shift : 32 + shift, shift : 32 + shift, shift : 32 + shift
        ],
        name="nuclei",
    )
    viewer.add_labels(
        labels[shift : 32 + shift, shift : 32 + shift, shift : 32 + shift]
    )
    [
        viewer.add_image(
            channel[shift : 32 + shift, shift : 32 + shift, shift : 32 + shift]
        )
        for channel in extra_channels
    ]

    labeled_channels = extract_labels_from_channels(
        labels[shift : 32 + shift, shift : 32 + shift, shift : 32 + shift],
        [
            c[shift : 32 + shift, shift : 32 + shift, shift : 32 + shift]
            for c in extra_channels
        ],
        radius=4,
        viewer=viewer,
    )
    table = pd.DataFrame(
        labeled_channels.items(), columns=["name", "pixels count"]
    )
    print(table)
    # [viewer.add_labels(item, name=key) for key, item in labeled_channels.items()]
    # expanded = expand_labels(labels, 4)
    # viewer.add_labels(expanded)
    napari.run()
