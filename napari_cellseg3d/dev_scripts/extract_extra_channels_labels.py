import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import expand_labels
from tqdm import tqdm


def extract_labels_from_channels(  # TODO add separate channels results
    nuclei_labels: np.array,
    extra_channels: list,
    radius: int = 4,
    threshold_factor=2,
    viewer=None,
):
    """
    Attemps to extract labels from other channels by expanding nuclei labels and picking the one with most pixels around it.
    Args:
        nuclei_labels (np.array): labels for the nuclei
        extra_channels (list): channels arrays to extract labels from
        radius: radius in which the approximation is made

    Returns:
    A list of extracted labels for each extra channel
    """
    labeled_channels = []
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
    for label_id in tqdm(np.unique(nuclei_labels)):
        if label_id == 0:
            continue
        label_nucleus = np.where(nuclei_labels == label_id, nuclei_labels, 0)
        expanded = expand_labels(label_nucleus, distance=radius)
        restricted = np.where(expanded != 0, nuclei_labels, 0)
        overlap = np.where(restricted != label_id, restricted, 0)

        for i, channel in enumerate(contrasted_channels):
            label_contrasted = np.where(expanded != 0, channel, 0)
            if overlap.any() != 0:
                max_labeled = 0
                for overlap_id in np.unique(overlap):
                    if overlap_id == 0:
                        continue
                    assigned_pixels = np.count_nonzero(
                        np.where(overlap == overlap_id, channel, 0)
                    )
                    if assigned_pixels > max_labeled:
                        max_labeled = assigned_pixels
                        max_label_id = overlap_id
                        if label_id != max_label_id:
                            labeled_channels.append(
                                np.zeros_like(label_contrasted)
                            )
            else:
                labeled_channel = np.where(label_contrasted != 0, label_id, 0)
                labeled_channels.append(labeled_channel)
                if (
                    np.count_nonzero(labeled_channel) > 0
                    and viewer is not None
                ):
                    viewer.add_labels(
                        labeled_channel, name=f"label_{label_id}_channel_{i+1}"
                    )

    cat_labels = np.zeros_like(nuclei_labels)
    for labels in np.unique(labeled_channels):
        if labels == 0:
            continue
        cat_labels += np.where(labels != 0, labels, 0)
    return cat_labels


if __name__ == "__main__":
    from pathlib import Path

    import napari
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

    viewer.add_labels(labeled_channels)
    # [viewer.add_labels(item, name=key) for key, item in labeled_channels.items()]
    # expanded = expand_labels(labels, 4)
    # viewer.add_labels(expanded)
    napari.run()
