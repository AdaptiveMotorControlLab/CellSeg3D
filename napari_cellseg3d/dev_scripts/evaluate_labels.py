import numpy as np
from collections import Counter
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from typing import Dict
import napari

from napari_cellseg3d.utils import LOGGER as log

PERCENT_CORRECT = 0.7

@dataclass
class LabelInfo:
    gt_index: int
    model_labels_id_and_status: Dict = None  # for each model label id present on gt_index in gt labels, contains status (correct/wrong)
    best_model_label_coverage: float = 0.0  # ratio of pixels of the gt label correctly labelled
    overall_gt_label_coverage: float = 0.0  # true positive ration of the model

    def get_correct_ratio(self):
        for model_label, status in self.model_labels_id_and_status.items():
            if status == "correct":
                return self.best_model_label_coverage
            else:
                return None

def eval_model(gt_labels, model_labels, print_report=False):

    report_list, new_labels, fused_labels = create_label_report(gt_labels, model_labels)

    per_label_perfs = []
    for report in report_list:
        if print_report:
            log.info(f"Label {report.gt_index} : {report.model_labels_id_and_status}")
            log.info(f"Best model label coverage : {report.best_model_label_coverage}")
            log.info(f"Overall gt label coverage : {report.overall_gt_label_coverage}")

        perf = report.get_correct_ratio()
        if perf is not None:
            per_label_perfs.append(perf)

    per_label_perfs = np.array(per_label_perfs)
    return per_label_perfs.mean(), new_labels, fused_labels




def create_label_report(gt_labels, model_labels):
    """Map the model's labels to the neurons labels.
    Parameters
    ----------
    gt_labels : ndarray
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
    map_fused_neurons = {}
    "background_labels contains all model labels where gt_labels is 0 and model_labels is not 0"
    background_labels = model_labels[np.where((gt_labels == 0))]
    "new_labels contains all labels in model_labels for which more than PERCENT_CORRECT% of the pixels are not labelled in gt_labels"
    new_labels = []
    for lab in np.unique(background_labels):
        if lab == 0:
            continue
        gt_background_size_at_lab = (
            gt_labels[np.where((model_labels == lab) & (gt_labels == 0))]
            .flatten()
            .shape[0]
        )
        gt_lab_size = (
            gt_labels[np.where(model_labels == lab)].flatten().shape[0]
        )
        if gt_background_size_at_lab / gt_lab_size > PERCENT_CORRECT:
            new_labels.append(lab)

    label_report_list = []
    # label_report = {}  # contains a dict saying which labels are correct or wrong for each gt label
    # model_label_values = {}  # contains the model labels value assigned to each unique gt label
    not_found_id = 0

    for i in tqdm(np.unique(gt_labels)):
        if i == 0:
            continue

        gt_label = gt_labels[np.where(gt_labels == i)]  # get a single gt label

        model_lab_on_gt = model_labels[
            np.where(((gt_labels == i) & (model_labels != 0)))
        ]  # all models labels on single gt_label
        info = LabelInfo(i)

        info.model_labels_id_and_status = {
            label_id: "" for label_id in np.unique(model_lab_on_gt)
        }

        if model_lab_on_gt.shape[0] == 0:
            info.model_labels_id_and_status[
                f"not_found_{not_found_id}"
            ] = "not found"
            not_found_id += 1
            label_report_list.append(info)
            continue

        log.debug(f"model_lab_on_gt : {np.unique(model_lab_on_gt)}")

        #  create LabelInfo object and init model_labels_id_and_status with all unique model labels on gt_label
        log.debug(
            f"info.model_labels_id_and_status : {info.model_labels_id_and_status}"
        )

        ratio = []
        for model_lab_id in info.model_labels_id_and_status.keys():
            size_model_label = (
                model_lab_on_gt[np.where(model_lab_on_gt == model_lab_id)]
                .flatten()
                .shape[0]
            )
            size_gt_label = gt_label.flatten().shape[0]

            log.debug(f"size_model_label : {size_model_label}")
            log.debug(f"size_gt_label : {size_gt_label}")

            ratio.append(size_model_label / size_gt_label)

        # log.debug(ratio)
        ratio_model_lab_for_given_gt_lab = np.array(ratio)
        info.best_model_label_coverage = (
            ratio_model_lab_for_given_gt_lab.max()
        )

        best_model_lab_id = model_lab_on_gt[
            np.argmax(ratio_model_lab_for_given_gt_lab)
        ]
        log.debug(f"best_model_lab_id : {best_model_lab_id}")

        info.overall_gt_label_coverage = (
            ratio_model_lab_for_given_gt_lab.sum()
        )  # the ratio of the pixels of the true label correctly labelled

        if info.best_model_label_coverage > PERCENT_CORRECT:
            info.model_labels_id_and_status[best_model_lab_id] = "correct"
            # info.model_labels_id_and_size[best_model_lab_id] = model_labels[np.where(model_labels == best_model_lab_id)].flatten().shape[0]
        else:
            info.model_labels_id_and_status[best_model_lab_id] = "wrong"
        for model_lab_id in np.unique(model_lab_on_gt):
            if model_lab_id != best_model_lab_id:
                log.debug(model_lab_id, "is wrong")
                info.model_labels_id_and_status[model_lab_id] = "wrong"

        label_report_list.append(info)

    correct_labels_id = []
    for report in label_report_list:
        for i_lab in report.model_labels_id_and_status.keys():
            if report.model_labels_id_and_status[i_lab] == "correct":
                correct_labels_id.append(i_lab)
    """Find all labels in label_report_list that are correct more than once"""
    duplicated_labels = [
        item for item, count in Counter(correct_labels_id).items() if count > 1
    ]
    "Sum up the size of all duplicated labels"
    for i in duplicated_labels:
        for report in label_report_list:
            if (
                i in report.model_labels_id_and_status.keys()
                and report.model_labels_id_and_status[i] == "correct"
            ):
                size = (
                    model_labels[np.where(model_labels == i)]
                    .flatten()
                    .shape[0]
                )
                map_fused_neurons[i] = size

    return label_report_list, new_labels, map_fused_neurons


def map_labels(gt_labels, model_labels):
    """Map the model's labels to the neurons labels.
    Parameters
    ----------
    gt_labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    Returns
    -------
    map_labels_existing: numpy array
        The label value of the model and the label value of the neuron associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label correctly labelled
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
        indexes = gt_labels[model_labels == i]
        # find the most common labels in the label i of the model
        unique, counts = np.unique(indexes, return_counts=True)
        tmp_map = []
        total_pixel_found = 0

        # log.debug(f"i: {i}")
        for ii in range(len(unique)):
            true_positive_ratio_model = counts[ii] / np.sum(counts)
            # if >50% of the pixels of the label i of the model correspond to the background it is considered as an artifact, that should not have been found
            # log.debug(f"unique: {unique[ii]}")
            if unique[ii] == 0:
                if true_positive_ratio_model > 0.5:
                    # -> artifact found
                    new_labels.append([i, true_positive_ratio_model])
            else:
                # if >50% of the pixels of the label unique[ii] of the true label map to the same label i of the model,
                # the label i is considered either as a fused neurons, if it the case for multiple unique[ii] or as neurone found
                ratio_pixel_found = counts[ii] / np.sum(
                    gt_labels == unique[ii]
                )
                if ratio_pixel_found > 0.8:
                    total_pixel_found += np.sum(counts[ii])
                    tmp_map.append(
                        [
                            i,
                            unique[ii],
                            ratio_pixel_found,
                            true_positive_ratio_model,
                        ]
                    )

        if len(tmp_map) == 1:
            # map to only one true neuron -> found neuron
            map_labels_existing.append(tmp_map[0])
        elif len(tmp_map) > 1:
            # map to multiple true neurons -> fused neuron
            for ii in range(len(tmp_map)):
                if total_pixel_found > np.sum(counts):
                    raise ValueError(
                        f"total_pixel_found > np.sum(counts) : {total_pixel_found} > {np.sum(counts)}"
                    )
                tmp_map[ii][3] = total_pixel_found / np.sum(counts)
            map_fused_neurons += tmp_map

    # log.debug(f"map_labels_existing: {map_labels_existing}")
    # log.debug(f"map_fused_neurons: {map_fused_neurons}")
    # log.debug(f"new_labels: {new_labels}")
    return map_labels_existing, map_fused_neurons, new_labels


def evaluate_model_performance(
    labels, model_labels, do_print=False, visualize=False
):
    """Evaluate the model performance.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    do_print : bool
        If True, print the results.
    visualize : bool
        If True, visualize the results.
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
        mean_true_positive_ratio_model = np.mean(
            [i[3] for i in map_labels_existing]
        )
        # calculate the mean ratio of the neurons pixels correctly labelled
        mean_ratio_pixel_found = np.mean([i[2] for i in map_labels_existing])
    else:
        mean_true_positive_ratio_model = np.nan
        mean_ratio_pixel_found = np.nan

    if len(map_fused_neurons) > 0:
        # calculate the mean ratio of the neurons pixels correctly labelled for the fused neurons
        mean_ratio_pixel_found_fused = np.mean(
            [i[2] for i in map_fused_neurons]
        )
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
        log.info("Neurons found: ")
        log.info(neurons_found)
        log.info("Neurons fused: ")
        log.info(neurons_fused)
        log.info("Neurons not found: ")
        log.info(neurons_not_found)
        log.info("Artefacts found: ")
        log.info(artefacts_found)
        log.info(
            "Mean true positive ratio of the model: ",
        )
        log.info(mean_true_positive_ratio_model)
        log.info(
            "Mean ratio of the neurons pixels correctly labelled: ",
        )
        log.info(mean_ratio_pixel_found)
        log.info(
            "Mean ratio of the neurons pixels correctly labelled for fused neurons: ",
        )
        log.info(mean_ratio_pixel_found_fused)
        log.info(
            "Mean true positive ratio of the model for fused neurons: ",
        )
        log.info(mean_true_positive_ratio_model_fused)
        log.info(
            "Mean ratio of false pixel in artefacts: "
        )
        log.info(mean_ratio_false_pixel_artefact)

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
                np.isin(unique_labels, neurons_found_labels) == False,
                unique_labels,
                0,
            )
            neurones_not_found_labels = neurones_not_found_labels[
                neurones_not_found_labels != 0
            ]
            not_found = np.where(
                np.isin(labels, neurones_not_found_labels), labels, 0
            )
            viewer.add_labels(not_found, name="ground truth not found")
            artefacts_found = np.where(
                np.isin(model_labels, [i[0] for i in new_labels]),
                model_labels,
                0,
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
    log.debug(np.array(results).shape)
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
