import numpy as np
import pandas as pd
from tqdm import tqdm
import napari

from napari_cellseg3d.utils import LOGGER as log
def map_labels(labels, model_labels):
    """Map the model's labels to the neurons labels.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    Returns
    -------
    map_labels_existing: numpy array
        The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label correctly labelled
    map_fused_neurons: numpy array
        The neurones are considered fused if they are labelled by the same model's label, in this case we will return The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label that are in one of the fused neurones
    new_labels: list
        The labels of the model that are not labelled in the neurons, the ratio of the pixels of the model's label that are an artefact
    """
    map_labels_existing = []
    map_fused_neurons = []
    new_labels = []

    for i in tqdm(np.unique(model_labels)):
        if i == 0:
            continue
        indexes = labels[model_labels == i]
        # find the most common labels in the label i of the model
        unique, counts = np.unique(indexes, return_counts=True)
        tmp_map = []
        total_pixel_found = 0
        for ii in range(len(unique)):
            true_positive_ratio_model = counts[ii] / np.sum(counts)
            # if >50% of the pixels of the label i of the model correspond to the background it is considered as an artifact, that should not have been found
            log.debug(f"unique: {unique[ii]}")
            if unique[ii] == 0:
                if true_positive_ratio_model > 0.5:
                    # -> artifact found
                    new_labels.append([i, true_positive_ratio_model])
            else:
                # if >50% of the pixels of the label unique[ii] of the true label map to the same label i of the model,
                # the label i is considered either as a fused neurons, if it the case for multiple unique[ii] or as neurone found
                ratio_pixel_found = counts[ii] / np.sum(labels == unique[ii])
                if ratio_pixel_found > 0.8:
                    total_pixel_found += np.sum(counts[ii])
                    tmp_map.append(
                        [i, unique[ii], ratio_pixel_found, true_positive_ratio_model]
                    )
                if total_pixel_found > np.sum(counts):
                    raise ValueError(f"total_pixel_found > np.sum(counts) : {total_pixel_found} > {np.sum(counts)}")

        if len(tmp_map) == 1:
            # map to only one true neuron -> found neuron
            map_labels_existing.append(tmp_map[0])
        elif len(tmp_map) > 1:
            # map to multiple true neurons -> fused neuron
            for ii in range(len(tmp_map)):
                # if total_pixel_found > np.sum(counts):
                #     raise ValueError(
                #         f"total_pixel_found > np.sum(counts[ii]) : {total_pixel_found} > {np.sum(counts)}"
                #     )
                tmp_map[ii][3] = total_pixel_found / np.sum(counts)
            map_fused_neurons += tmp_map
    return map_labels_existing, map_fused_neurons, new_labels


def evaluate_model_performance(labels, model_labels, do_print=True, visualize=False):
    """Evaluate the model performance.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    do_print : bool
        If True, print the results.
    Returns
    -------
    neuron_found : float
        The number of neurons found by the model
    neuron_fused: float
        The number of neurons fused by the model
    neuron_not_found: float
        The number of neurons not found by the model
    neuron_artefact: float
        The number of artefact that the model wrongly labelled as neurons
    mean_true_positive_ratio_model: float
        The mean (over the model's labels that correspond to one true label) of (correctly labelled pixels)/(total number of pixels of the model's label)
    mean_ratio_pixel_found: float
        The mean (over the model's labels that correspond to one true label) of (correctly labelled pixels)/(total number of pixels of the true label)
    mean_ratio_pixel_found_fused: float
        The mean (over the model's labels that correspond to multiple true label) of (correctly labelled pixels)/(total number of pixels of the true label)
    mean_true_positive_ratio_model_fused: float
        The mean (over the model's labels that correspond to multiple true label) of (correctly labelled pixels in any fused neurons of this model's label)/(total number of pixels of the model's label)
    mean_ratio_false_pixel_artefact: float
        The mean (over the model's labels that are not labelled in the neurons) of (wrongly labelled pixels)/(total number of pixels of the model's label)
    """
    log.debug("Mapping labels...")
    map_labels_existing, map_fused_neurons, new_labels = map_labels(
        labels, model_labels
    )

    # calculate the number of neurons individually found
    neurons_found = len(map_labels_existing)
    # calculate the number of neurons fused
    neurons_fused = len(map_fused_neurons)
    # calculate the number of neurons not found
    log.debug("Calculating the number of neurons not found...")
    neurons_found_labels = np.unique(
        [i[1] for i in map_labels_existing] + [i[1] for i in map_fused_neurons]
    )
    unique_labels = np.unique(labels)
    neurons_not_found = len(unique_labels) - 1 - len(neurons_found_labels)
    # artefacts found
    artefacts_found = len(new_labels)
    if len(map_labels_existing) > 0:
        # calculate the mean true positive ratio of the model
        mean_true_positive_ratio_model = np.mean([i[3] for i in map_labels_existing])
        # calculate the mean ratio of the neurons pixels correctly labelled
        mean_ratio_pixel_found = np.mean([i[2] for i in map_labels_existing])
    else:
        mean_true_positive_ratio_model = np.nan
        mean_ratio_pixel_found = np.nan

    if len(map_fused_neurons) > 0:
        # calculate the mean ratio of the neurons pixels correctly labelled for the fused neurons
        mean_ratio_pixel_found_fused = np.mean([i[2] for i in map_fused_neurons])
        # calculate the mean true positive ratio of the model for the fused neurons
        mean_true_positive_ratio_model_fused = np.mean(
            [i[3] for i in map_fused_neurons]
        )
    else:
        mean_ratio_pixel_found_fused = np.nan
        mean_true_positive_ratio_model_fused = np.nan

    # calculate the mean false positive ratio of each artefact
    if len(new_labels) > 0:
        mean_ratio_false_pixel_artefact = np.mean([i[1] for i in new_labels])
    else:
        mean_ratio_false_pixel_artefact = np.nan

    if do_print:
        print("Neurons found: ", neurons_found)
        print("Neurons fused: ", neurons_fused)
        print("Neurons not found: ", neurons_not_found)
        print("Artefacts found: ", artefacts_found)
        print("Mean true positive ratio of the model: ", mean_true_positive_ratio_model)
        print(
            "Mean ratio of the neurons pixels correctly labelled: ",
            mean_ratio_pixel_found,
        )
        print(
            "Mean ratio of the neurons pixels correctly labelled for fused neurons: ",
            mean_ratio_pixel_found_fused,
        )
        print(
            "Mean true positive ratio of the model for fused neurons: ",
            mean_true_positive_ratio_model_fused,
        )
        print(
            "Mean ratio of false pixel in artefacts: ", mean_ratio_false_pixel_artefact
        )
        if visualize:
            viewer = napari.Viewer()
            viewer.add_labels(labels, name="ground truth")
            viewer.add_labels(model_labels, name="model's labels")
            found_model = np.where(
                np.isin(model_labels, [i[0] for i in map_labels_existing]),
                model_labels,
                0,
            )
            viewer.add_labels(found_model, name="model's labels found")
            found_label = np.where(
                np.isin(labels, [i[1] for i in map_labels_existing]), labels, 0
            )
            viewer.add_labels(found_label, name="ground truth found")
            neurones_not_found_labels = np.where(
                np.isin(unique_labels, neurons_found_labels) == False, unique_labels, 0
            )
            neurones_not_found_labels = neurones_not_found_labels[
                neurones_not_found_labels != 0
                ]
            not_found = np.where(np.isin(labels, neurones_not_found_labels), labels, 0)
            viewer.add_labels(not_found, name="ground truth not found")
            artefacts_found = np.where(
                np.isin(model_labels, [i[0] for i in new_labels]), model_labels, 0
            )
            viewer.add_labels(artefacts_found, name="model's labels artefacts")
            fused_model = np.where(
                np.isin(model_labels, [i[0] for i in map_fused_neurons]),
                model_labels,
                0,
            )
            viewer.add_labels(fused_model, name="model's labels fused")
            fused_label = np.where(
                np.isin(labels, [i[1] for i in map_fused_neurons]), labels, 0
            )
            viewer.add_labels(fused_label, name="ground truth fused")
            napari.run()

    return (
        neurons_found,
        neurons_fused,
        neurons_not_found,
        artefacts_found,
        mean_true_positive_ratio_model,
        mean_ratio_pixel_found,
        mean_ratio_pixel_found_fused,
        mean_true_positive_ratio_model_fused,
        mean_ratio_false_pixel_artefact,
    )


def save_as_csv(results, path):
    """
    Save the results as a csv file

    Parameters
    ----------
    results: list
        The results of the evaluation
    path: str
        The path to save the csv file
    """
    print(np.array(results).shape)
    df = pd.DataFrame(
        [results],
        columns=[
            "neurons_found",
            "neurons_fused",
            "neurons_not_found",
            "artefacts_found",
            "mean_true_positive_ratio_model",
            "mean_ratio_pixel_found",
            "mean_ratio_pixel_found_fused",
            "mean_true_positive_ratio_model_fused",
            "mean_ratio_false_pixel_artefact",
        ],
    )
    df.to_csv(path, index=False)


# if __name__ == "__main__":
#     """
#     # Example of how to use the functions in this module.
#     a = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
#
#     b = np.array([[5, 5, 0, 0], [5, 5, 2, 0], [0, 2, 2, 0], [0, 0, 2, 0]])
#     evaluate_model_performance(a, b)
#
#     c = np.array([[2, 2, 0, 0], [2, 2, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
#
#     d = np.array([[4, 0, 4, 0], [4, 4, 4, 0], [0, 4, 4, 0], [0, 0, 4, 0]])
#
#     evaluate_model_performance(c, d)
#
#     from tifffile import imread
#     labels=imread("dataset/visual_tif/labels/testing_im_new_label.tif")
#     labels_model=imread("dataset/visual_tif/artefact_neurones/basic_model.tif")
#     evaluate_model_performance(labels, labels_model,visualize=True)
#     """
#     from tifffile import imread
#
#     labels = imread("dataset_clean/VALIDATION/validation_labels.tif")
#     try:
#         labels_model = imread("results/watershed_based_model/instance_labels.tif")
#     except:
#         raise Exception(
#             "you should download the model's label that are under results (output and statistics)/watershed_based_model/instance_labels.tif and put it in the folder results/watershed_based_model/"
#         )
#
#     evaluate_model_performance(labels, labels_model, visualize=True)
