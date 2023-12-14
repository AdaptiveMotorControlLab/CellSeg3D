import threading
import time
import warnings
from functools import partial
from pathlib import Path

import napari
import numpy as np
import scipy.ndimage as ndimage
from napari.qt.threading import thread_worker
from tifffile import imread, imwrite
from tqdm import tqdm

import napari_cellseg3d.dev_scripts.artefact_labeling as make_artefact_labels
from napari_cellseg3d.code_models.instance_segmentation import binary_watershed

# import sys
# sys.path.append(str(Path(__file__) / "../../"))


"""
New code by Yves Paychère
Fixes labels and allows to auto-detect artifacts and neurons based on a simple intenstiy threshold
"""


def relabel_non_unique_i(label, save_path, go_fast=False):
    """Relabel the image labelled with different label for each neuron and save it in the save_path location.

    Parameters
    ----------
    label : np.array
        the label image
    save_path : str
        the path to save the relabeld image.
    """
    value_label = 0
    new_labels = np.zeros_like(label)
    map_labels_existing = []
    unique_label = np.unique(label)
    for i_label in tqdm(
        range(len(unique_label)), desc="relabeling", ncols=100
    ):
        i = unique_label[i_label]
        if i == 0:
            continue
        if go_fast:
            new_label, to_add = ndimage.label(label == i)
            map_labels_existing.append(
                [i, list(range(value_label + 1, value_label + to_add + 1))]
            )

        else:
            # catch the warning of the watershed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_label = binary_watershed(label == i)
                unique = np.unique(new_label)
                to_add = unique[-1]
                map_labels_existing.append([i, unique[1:] + value_label])

        new_label[new_label != 0] += value_label
        new_labels += new_label
        value_label += to_add

    imwrite(save_path, new_labels)
    return map_labels_existing


def add_label(old_label, artefact, new_label_path, i_labels_to_add):
    """Add the label to the label image.

    Parameters
    ----------
    old_label : np.array
        the label image
    artefact : np.array
        the artefact image that contains some neurons
    new_label_path : str
        the path to save the new label image.
    """
    new_label = old_label.copy()
    max_label = np.max(old_label)
    for i, i_label in enumerate(i_labels_to_add):
        new_label[artefact == i_label] = i + max_label + 1
    imwrite(new_label_path, new_label)


returns = []


def ask_labels(unique_artefact, test=False):
    global returns
    returns = []
    if not test:
        i_labels_to_add_tmp = input(
            "Which labels do you want to add (0 to skip) ? (separated by a comma):"
        )
        i_labels_to_add_tmp = [int(i) for i in i_labels_to_add_tmp.split(",")]
    else:
        i_labels_to_add_tmp = [0]

    if i_labels_to_add_tmp == [0]:
        print("no label added")
        returns = [[]]
        print("close the napari window to continue")
        return

    for i in i_labels_to_add_tmp:
        if i == 0:
            print("0 is not a valid label")
            # delete the 0
            i_labels_to_add_tmp.remove(i)
    # test if all index are negative
    if all(i < 0 for i in i_labels_to_add_tmp):
        print(
            "all labels are negative-> will add all the labels except the one you gave"
        )
        i_labels_to_add = list(unique_artefact)
        for i in i_labels_to_add_tmp:
            if np.abs(i) in i_labels_to_add:
                i_labels_to_add.remove(np.abs(i))
            else:
                print("the label", np.abs(i), "is not in the label image")
        i_labels_to_add_tmp = i_labels_to_add
    else:
        # remove the negative index
        for i in i_labels_to_add_tmp:
            if i < 0:
                i_labels_to_add_tmp.remove(i)
                print(
                    "ignore the negative label",
                    i,
                    " since not all the labels are negative",
                )
            if i not in unique_artefact:
                print("the label", i, "is not in the label image")
                i_labels_to_add_tmp.remove(i)

    returns = [i_labels_to_add_tmp]
    print("close the napari window to continue")


def relabel(
    image_path,
    label_path,
    go_fast=False,
    check_for_unicity=True,
    delay=0.3,
    viewer=None,
    test=False,
):
    """Relabel the image labelled with different label for each neuron and save it in the save_path location.

    Parameters
    ----------
    image_path : str
        the path to the image
    label_path : str
        the path to the label image
    go_fast : bool, optional
        if True, the relabeling will be faster but the labels can more frequently be merged, by default False
    check_for_unicity : bool, optional
        if True, the relabeling will check if the labels are unique, by default True
    delay : float, optional
        the delay between each image for the visualization, by default 0.3
    viewer : napari.Viewer, optional
        the napari viewer, by default None.
    """
    global returns

    label = imread(label_path)
    initial_label_path = label_path
    if check_for_unicity:
        # check if the label are unique
        new_label_path = label_path[:-4] + "_relabel_unique.tif"
        map_labels_existing = relabel_non_unique_i(
            label, new_label_path, go_fast=go_fast
        )
        print(
            "visualize the relabeld image in white the previous labels and in red the new labels"
        )
        if not test:
            visualize_map(
                map_labels_existing, label_path, new_label_path, delay=delay
            )
        label_path = new_label_path
    # detect artefact
    print("detection of potential neurons (in progress)")
    image = imread(image_path)
    artefact = make_artefact_labels.make_artefact_labels(
        image,
        imread(label_path),
        do_multi_label=True,
        threshold_artefact_brightness_percent=30,
        threshold_artefact_size_percent=0,
        contrast_power=30,
    )
    print("detection of potential neurons (done)")
    # ask the user if the artefact are not neurons
    i_labels_to_add = []
    loop = True
    unique_artefact = list(np.unique(artefact))
    while loop:
        # visualize the artefact and ask the user which label to add to the label image
        t = threading.Thread(
            target=partial(ask_labels, test=test), args=(unique_artefact,)
        )
        t.start()
        artefact_copy = np.where(
            np.isin(artefact, i_labels_to_add), 0, artefact
        )
        if viewer is None:
            viewer = napari.view_image(image)
        else:
            viewer = viewer
            viewer.add_image(image, name="image")
        viewer.add_labels(artefact_copy, name="potential neurons")
        viewer.add_labels(imread(label_path), name="labels")
        if not test:
            napari.run()
        t.join()
        i_labels_to_add_tmp = returns[0]
        # check if the selected labels are neurones
        for i in i_labels_to_add:
            if i not in i_labels_to_add_tmp:
                i_labels_to_add_tmp.append(i)
        artefact_copy = np.where(
            np.isin(artefact, i_labels_to_add_tmp), artefact, 0
        )
        print("these labels will be added")
        if test:
            viewer.close()
        viewer = napari.view_image(image) if viewer is None else viewer
        if not test:
            viewer.add_labels(artefact_copy, name="labels added")
            napari.run()
            revert = input("Do you want to revert? (y/n)")
        if test:
            revert = "n"
            viewer.close()
        if revert != "y":
            i_labels_to_add = i_labels_to_add_tmp
            for i in i_labels_to_add:
                if i in unique_artefact:
                    unique_artefact.remove(i)
        if test:
            break
        loop = input("Do you want to add more labels? (y/n)") == "y"
    # add the label to the label image
    new_label_path = initial_label_path[:-4] + "_new_label.tif"
    print("the new label will be saved in", new_label_path)
    add_label(imread(label_path), artefact, new_label_path, i_labels_to_add)
    # store the artefact remaining
    new_artefact_path = initial_label_path[:-4] + "_artefact.tif"
    artefact = np.where(np.isin(artefact, i_labels_to_add), 0, artefact)
    imwrite(new_artefact_path, artefact)


def modify_viewer(old_label, new_label, args):
    """Modify the viewer to show the relabeling.

    Parameters
    ----------
    old_label : napari.layers.Labels
        the layer of the old label
    new_label : napari.layers.Labels
        the layer of the new label
    args : list
        the first element is the old label and the second element is the new label.
    """
    if args == "hide new label":
        new_label.visible = False
    elif args == "show new label":
        new_label.visible = True
    else:
        old_label.selected_label = args[0]
        if not np.isnan(args[1]):
            new_label.selected_label = args[1]


@thread_worker
def to_show(map_labels_existing, delay=0.5):
    """Modify the viewer to show the relabeling.

    Parameters
    ----------
    map_labels_existing : list
        the list of the of the map between the old label and the new label
    delay : float, optional
        the delay between each image for the visualization, by default 0.3.
    """
    time.sleep(2)
    for i in map_labels_existing:
        yield "hide new label"
        if len(i[1]):
            yield [i[0], i[1][0]]
        else:
            yield [i[0], np.nan]
        time.sleep(delay)
        yield "show new label"
        for j in i[1]:
            yield [i[0], j]
            time.sleep(delay)


def create_connected_widget(
    old_label, new_label, map_labels_existing, delay=0.5
):
    """Builds a widget that can control a function in another thread."""
    worker = to_show(map_labels_existing, delay)
    worker.start()
    worker.yielded.connect(
        lambda arg: modify_viewer(old_label, new_label, arg)
    )


def visualize_map(map_labels_existing, label_path, relabel_path, delay=0.5):
    """Visualize the map of the relabeling.

    Parameters
    ----------
    map_labels_existing : list
        the list of the relabeling.
    """
    label = imread(label_path)
    relabel = imread(relabel_path)

    viewer = napari.Viewer(ndisplay=3)

    old_label = viewer.add_labels(label, num_colors=3)
    new_label = viewer.add_labels(relabel, num_colors=3)
    old_label.colormap.colors = np.array(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    new_label.colormap.colors = np.array(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
    )

    # viewer.dims.ndisplay = 3
    viewer.camera.angles = (180, 3, 50)
    viewer.camera.zoom = 1

    old_label.show_selected_label = True
    new_label.show_selected_label = True

    create_connected_widget(
        old_label, new_label, map_labels_existing, delay=delay
    )
    napari.run()


def relabel_non_unique_i_folder(folder_path, end_of_new_name="relabeled"):
    """Relabel the image labelled with different label for each neuron and save it in the save_path location.

    Parameters
    ----------
    folder_path : str
        the path to the folder containing the label images
    end_of_new_name : str
        thename to add at the end of the relabled image.
    """
    for file in Path.iterdir(folder_path):
        if file.suffix == ".tif":
            label = imread(str(Path(folder_path / file)))
            relabel_non_unique_i(
                label,
                str(Path(folder_path / file[:-4] + end_of_new_name + ".tif")),
            )


if __name__ == "__main__":
    im_path = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_DATA/visual"

    # i = 4
    # im_id = i+1
    image_path = str(im_path / "visual.tif")
    gt_labels_path = str(im_path / "visual_gt.tif")
    relabel(image_path, gt_labels_path, check_for_unicity=True, go_fast=False)
