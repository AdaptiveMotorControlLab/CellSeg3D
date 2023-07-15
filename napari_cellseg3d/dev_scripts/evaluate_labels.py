import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd

from napari_cellseg3d.utils import LOGGER as log

PERCENT_CORRECT = 0.5  # how much of the original label should be found by the model to be classified as correct


def evaluate_model_performance(
    labels,
    model_labels,
    threshold_correct=PERCENT_CORRECT,
    print_details=False,
    visualize=False,
    return_graphical_summary=False,
):
    """Evaluate the model performance.
    Parameters
    ----------
    labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    print_details : bool
        If True, print the results.
    visualize : bool
        If True, visualize the results.
    return_graphical_summary : bool
        If True, return the distribution of the true positive, false positive and fused neurons depending on the test function.
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
    graph_true_positive_ratio_model: ndarray
    """
    log.debug("Mapping labels...")
    tmp = map_labels(
        labels,
        model_labels,
        threshold_correct,
        return_total_number_gt_labels=True,
        return_dict_map=True,
        return_graphical_summary=return_graphical_summary,
    )
    if return_graphical_summary:
        (
            map_labels_existing,
            map_fused_neurons,
            map_multiple_label,
            new_labels,
            n_gt,
            dict_map,
            graph_true_positive_ratio_model,
        ) = tmp
    else:
        (
            map_labels_existing,
            map_fused_neurons,
            map_multiple_label,
            new_labels,
            n_gt,
            dict_map,
        ) = tmp

    # calculate the number of neurons individually found
    neurons_found = len(map_labels_existing)
    # calculate the number of neurons fused
    neurons_fused = len(map_fused_neurons)
    # calculate the number of neurons not found
    log.debug("Calculating the number of neurons not found...")
    tmp = np.array([])
    if len(map_labels_existing):
        tmp = map_labels_existing[:, dict_map["gt_label"]].astype(int)
    if len(map_fused_neurons):
        tmp = np.concatenate(
            (
                tmp,
                map_fused_neurons[:, dict_map["gt_label"]].astype(int),
            ),
            axis=0,
        )
    if len(map_multiple_label):
        tmp = np.concatenate(
            (
                tmp,
                map_multiple_label[:, dict_map["gt_label"]].astype(int),
            )
        )
    n_found = np.sum(np.bincount(tmp) > 0)

    neurons_not_found = n_gt - n_found
    # artefacts found
    artefacts_found = len(new_labels)
    if len(map_labels_existing):
        # calculate the mean true positive ratio of the model
        mean_true_positive_ratio_model = np.mean(
            map_labels_existing[:, dict_map["ratio_model_label"]]
        )
        # calculate the mean ratio of the neurons pixels correctly labelled
        mean_ratio_pixel_found = np.mean(
            map_labels_existing[:, dict_map["ratio_tested"]]
        )
    else:
        mean_true_positive_ratio_model = np.nan
        mean_ratio_pixel_found = np.nan

    if len(map_fused_neurons):
        # calculate the mean ratio of the neurons pixels correctly labelled for the fused neurons
        mean_ratio_pixel_found_fused = np.mean(
            map_fused_neurons[:, dict_map["ratio_tested"]]
        )
        # calculate the mean true positive ratio of the model for the fused neurons
        mean_true_positive_ratio_model_fused = np.mean(
            map_fused_neurons[:, dict_map["ratio_model_label"]]
        )
    else:
        mean_ratio_pixel_found_fused = np.nan
        mean_true_positive_ratio_model_fused = np.nan

    # calculate the mean false positive ratio of each artefact
    if len(new_labels):
        mean_ratio_false_pixel_artefact = np.mean(new_labels[:, 1])
    else:
        mean_ratio_false_pixel_artefact = np.nan

    log.info(
        f"Percent of non-fused neurons found: {neurons_found / n_gt * 100:.2f}%"
    )
    log.info(
        f"Percent of fused neurons found: {neurons_fused / n_gt * 100:.2f}%"
    )
    log.info(
        f"Overall percent of neurons found: {(neurons_found + neurons_fused) / n_gt * 100:.2f}%"
    )

    if print_details:
        log.info(f"Neurons found: {neurons_found}")
        log.info(f"Neurons fused: {neurons_fused}")
        log.info(f"Neurons not found: {neurons_not_found}")
        log.info(f"Artefacts found: {artefacts_found}")
        log.info(
            f"Mean true positive ratio of the model: {mean_true_positive_ratio_model}"
        )
        log.info(
            f"Mean ratio of the neurons pixels correctly labelled: {mean_ratio_pixel_found}"
        )
        log.info(
            f"Mean ratio of the neurons pixels correctly labelled for fused neurons: {mean_ratio_pixel_found_fused}"
        )
        log.info(
            f"Mean true positive ratio of the model for fused neurons: {mean_true_positive_ratio_model_fused}"
        )
        log.info(
            f"Mean ratio of the false pixels labelled as neurons: {mean_ratio_false_pixel_artefact}"
        )

    if visualize:
        unique_labels = np.unique(labels)
        tmp = np.array([])
        if len(map_labels_existing):
            tmp = map_labels_existing[:, 1].astype(int)
        if len(map_fused_neurons):
            tmp = np.concatenate((tmp, map_fused_neurons[:, 1].astype(int)))
        if len(map_multiple_label):
            tmp = np.concatenate((tmp, map_multiple_label[:, 1].astype(int)))
        neurons_found_labels = np.unique(tmp)
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_labels(labels, name="ground truth")
        viewer.add_labels(model_labels, name="model's labels")
        found_model = np.where(
            np.isin(
                model_labels, map_labels_existing[:, dict_map["model_label"]]
            ),
            model_labels,
            0,
        )
        viewer.add_labels(found_model, name="model's labels found")
        found_label = np.where(
            np.isin(labels, map_labels_existing[:, dict_map["gt_label"]]),
            labels,
            0,
        )
        viewer.add_labels(found_label, name="ground truth found")
        neurones_not_found_labels = np.where(
            np.isin(unique_labels, neurons_found_labels) is False,
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
            np.isin(model_labels, new_labels[:, 0].astype(int)),
            model_labels,
            0,
        )
        viewer.add_labels(artefacts_found, name="model's labels artefacts")
        fused_model = np.where(
            np.isin(
                model_labels, map_fused_neurons[:, dict_map["model_label"]]
            ),
            model_labels,
            0,
        )
        viewer.add_labels(fused_model, name="model's labels fused")
        fused_label = np.where(
            np.isin(labels, map_fused_neurons[:, dict_map["gt_label"]]),
            labels,
            0,
        )
        viewer.add_labels(fused_label, name="ground truth fused")
        napari.run()
    res = (
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
    if return_graphical_summary:
        res = res + (graph_true_positive_ratio_model,)
    return res


def iou(intersection, gt_counts, model_counts):
    """Calculate the intersection over union (IoU) between the model's labels and the ground truth's labels.
    Parameters
    ----------
    intersection : ndarray
        The number of pixels of the model's labels that are correctly labelled.
    gt_counts : ndarray
        The number of pixels of the ground truth's labels.
    model_counts : ndarray
        The number of pixels of the model's labels.
    Returns
    -------
    iou: numpy array
        The intersection over union (IoU) between the model's labels and the ground truth's labels.
    """
    union = gt_counts + model_counts - intersection
    return intersection / union


def ioGroundTruth(intersection, gt_counts, model_counts=None):
    """Calculate the intersection over ground truth (IoGT) between the model's labels and the ground truth's labels.
    Parameters
    ----------
    intersection : ndarray
        The number of pixels of the model's labels that are correctly labelled.
    gt_counts : ndarray
        The number of pixels of the ground truth's labels.
    model_counts : ndarray
        The number of pixels of the model's labels.
    Returns
    -------
    iou: numpy array
        The intersection over ground truth (IoGT) between the model's labels and the ground truth's labels.
    """
    return intersection / gt_counts


def find_fusion(map_labels_existing, i_label, return_n_association=False):
    """Find the fused neurons that are labelled by the same label or neuron that is labelled by multiple labels.
    It is considered as a fusion if the same label precized by i_label appears multiple times in the map_labels_existing array.
    Parameters
    ----------
    map_labels_existing : ndarray
        The label value of the model and the label value of the neuron associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label correctly labelled
    i_label : int
        The index of the label in the map_labels_existing array.
    return_n_association : bool
        If True, return the number of association of the labels presized in i_label, sorted by the value of the label.
    Returns
    -------
    i_fused_neurones: numpy array
         a boolean array of the neurons that are fused
    n_association: numpy array
        The number of association of the label i_label sorted by the value of the label.
    """
    if len(map_labels_existing) < 2:
        if return_n_association:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.array([], dtype=int)
    _, i_inverse, n_mapped_to_same = np.unique(
        map_labels_existing[:, i_label],
        return_inverse=True,
        return_counts=True,
    )
    i_fused = n_mapped_to_same[i_inverse] > 1
    if return_n_association:
        return i_fused, n_mapped_to_same
    return i_fused


def map_labels(
    gt_labels,
    model_labels,
    threshold_correct=PERCENT_CORRECT,
    return_total_number_gt_labels=False,
    return_dict_map=False,
    accuracy_function=ioGroundTruth,
    return_graphical_summary=False,
):
    """Map the model's labels to the neurons labels.
    Parameters
    ----------
    gt_labels : ndarray
        Label image with neurons labelled as mulitple values.
    model_labels : ndarray
        Label image from the model labelled as mulitple values.
    threshold_correct : float
        The threshold of the ratio above which the model's label is considered as correctly labelled.
    return_total_number_gt_labels : bool
        If True, return the total number of ground truth's labels.
    return_dict_map : bool
        If True, return the dictionnary containing the index of the columns of the map_labels_existing array.
    accuracy_function : function
        The function used to calculate the accuracy of the model's labels.
    return_graphical_summary : bool
        If True, return the graphical summary of the mapping.
    Returns
    -------
    map_labels_existing: numpy array
        The label value of the model and the label value of the neuron associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label correctly labelled
    map_fused_neurons: numpy array
        The neurones are considered fused if they are labelled by the same model's label, in this case we will return The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label that are in one of the fused neurones
    map_multiple_labels: numpy array
        The neurones are considered as multiply labelled if they are labelled by multiple model's label, in this case we will return The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled by each label, the ratio of the pixels of the model's label that are in the neurone
    new_labels: list
        The labels of the model that are not labelled in the neurons, the ratio of the pixels of the model's label that are not in a known neurone.
    """
    counts_model_labels = np.bincount(model_labels.flatten())
    counts_gt_labels = np.bincount(gt_labels.flatten())
    # transformation to use np.unique to map the labels
    n_digit_model_labels = len(str(np.max(model_labels)))
    gt_labels = (
        gt_labels * 10**n_digit_model_labels
    )  # add 0 at the end of the gt labels to be able to concatenate them with the model labels
    concatened_labels = gt_labels + model_labels
    # get the unique combination of labels and their counts
    concatened_labels, counts = np.unique(
        concatened_labels, return_counts=True
    )
    # deconcatenate the labels respecting the order of counts
    gt_labels = concatened_labels // 10**n_digit_model_labels
    model_labels = concatened_labels % 10**n_digit_model_labels
    # remove the associations with the background of the model_labels
    counts = counts[model_labels != 0]
    gt_labels = gt_labels[model_labels != 0]
    model_labels = model_labels[model_labels != 0]

    n_gt_labels = np.array(
        [counts_gt_labels[gt_label] for gt_label in gt_labels]
    )
    n_model_labels = np.array(
        [counts_model_labels[model_label] for model_label in model_labels]
    )

    ratio_to_test = accuracy_function(counts, n_gt_labels, n_model_labels)
    ratio_model_labels = counts / n_model_labels
    dict_map = {
        "model_label": 0,
        "gt_label": 1,
        "ratio_tested": 2,
        "ratio_model_label": 3,
    }
    map_labels_existing = np.array(
        [
            [model_label, gt_label, ratio_tested, ratio_model_label]
            for model_label, gt_label, ratio_tested, ratio_model_label in zip(
                model_labels, gt_labels, ratio_to_test, ratio_model_labels
            )
            if gt_label != 0 and ratio_tested > threshold_correct
        ]
    )
    new_labels = np.array(
        [
            [model_label, ratio_model_label]
            for model_label, ratio_model_label, gt_label in zip(
                model_labels, ratio_model_labels, gt_labels
            )
            if gt_label == 0 and ratio_model_label > threshold_correct
        ]
    )

    # remove from new_labels the labels that are in map_labels_existing
    new_labels = np.array(new_labels)
    i_new_labels = np.isin(
        new_labels[:, dict_map["model_label"]],
        map_labels_existing[:, dict_map["model_label"]],
        invert=True,
    )
    new_labels = new_labels[i_new_labels, :]
    # find the fused neurons: multiple gt labels are mapped to the same model label
    i_fused_neurones, n_mapped_to_same_gt = find_fusion(
        map_labels_existing, dict_map["model_label"], return_n_association=True
    )
    map_fused_neurons = map_labels_existing[i_fused_neurones, :]
    map_labels_existing = map_labels_existing[~i_fused_neurones, :]
    # sum the ratio of the model's label that are in the fused neurons (n_mapped_to_same_gt is sorted by the value of the model's label)
    map_fused_neurons = map_fused_neurons[
        map_fused_neurons[:, dict_map["model_label"]].argsort()
    ]
    i = 0
    for n_mapped in n_mapped_to_same_gt[n_mapped_to_same_gt > 1]:
        map_fused_neurons[
            i : i + n_mapped, dict_map["ratio_model_label"]
        ] = np.sum(
            map_fused_neurons[i : i + n_mapped, dict_map["ratio_model_label"]]
        )
        i += n_mapped
    if PERCENT_CORRECT < 0.5:
        # find the multiple labelled neurons: one gt label is mapped to multiple model labels
        i_multiple_labelled_neurones = find_fusion(
            map_labels_existing, dict_map["gt_label"]
        )
        map_multiple_labelled_neurones = map_labels_existing[
            i_multiple_labelled_neurones, :
        ]
        map_labels_existing = map_labels_existing[
            ~i_multiple_labelled_neurones, :
        ]
    else:
        map_multiple_labelled_neurones = np.array([], dtype=float)
    to_return = (
        map_labels_existing,
        map_fused_neurons,
        map_multiple_labelled_neurones,
        new_labels,
    )
    if return_total_number_gt_labels:
        n_gt_labels = (
            np.sum(counts_gt_labels > 0) - 1
        )  # -1 to remove the background
        to_return = to_return + (n_gt_labels,)
    if return_dict_map:
        to_return = to_return + (dict_map,)
    if return_graphical_summary:
        # make a histogram of the ratio_to_test
        fig, ax = plt.subplots()
        unique_model_labels = np.unique(model_labels)
        best_association = np.zeros(unique_model_labels[-1] + 1)
        for model_label in unique_model_labels:
            best_association[model_label] = np.max(
                ratio_to_test[model_labels == model_label]
            )
        ax.hist(
            best_association[best_association > 0],
            bins=50,
            label="best association for each model's label",
            stacked=True,
        )
        to_plot = []
        labels = []
        if len(new_labels):
            to_plot.append(
                best_association[np.unique(new_labels[:, 0]).astype(int)]
            )
            labels.append("false positive")
        if len(map_labels_existing):
            to_plot.append(map_labels_existing[:, dict_map["ratio_tested"]])
            labels.append("true positive")
        if len(map_fused_neurons):
            to_plot.append(map_fused_neurons[:, dict_map["ratio_tested"]])
            labels.append("1 model label for multiple gt labels")
        if len(map_multiple_labelled_neurones):
            to_plot.append(
                map_multiple_labelled_neurones[:, dict_map["ratio_tested"]]
            )
            labels.append("multiple model labels for 1 gt label")
        ax.hist(
            to_plot,
            bins=50,
            label=labels,
            stacked=True,
        )
        ax.set_title(
            "distribution of the accuracy of the association between the model's labels and the gt labels"
        )
        ax.set_xlabel("accuracy")
        ax.set_ylabel("number of model's labels")
        ax.legend(loc="upper right")
        graphical_summary = fig
        to_return = to_return + (graphical_summary,)
    return to_return


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


#######################
# Slower version that was used for debugging
#######################

# from collections import Counter
# from dataclasses import dataclass
# from typing import Dict
# @dataclass
# class LabelInfo:
#     gt_index: int
#     model_labels_id_and_status: Dict = None  # for each model label id present on gt_index in gt labels, contains status (correct/wrong)
#     best_model_label_coverage: float = (
#         0.0  # ratio of pixels of the gt label correctly labelled
#     )
#     overall_gt_label_coverage: float = 0.0  # true positive ration of the model
#
#     def get_correct_ratio(self):
#         for model_label, status in self.model_labels_id_and_status.items():
#             if status == "correct":
#                 return self.best_model_label_coverage
#             else:
#                 return None


# def eval_model(gt_labels, model_labels, print_report=False):
#
#     report_list, new_labels, fused_labels = create_label_report(
#         gt_labels, model_labels
#     )
#     per_label_perfs = []
#     for report in report_list:
#         if print_report:
#             log.info(
#                 f"Label {report.gt_index} : {report.model_labels_id_and_status}"
#             )
#             log.info(
#                 f"Best model label coverage : {report.best_model_label_coverage}"
#             )
#             log.info(
#                 f"Overall gt label coverage : {report.overall_gt_label_coverage}"
#             )
#
#         perf = report.get_correct_ratio()
#         if perf is not None:
#             per_label_perfs.append(perf)
#
#     per_label_perfs = np.array(per_label_perfs)
#     return per_label_perfs.mean(), new_labels, fused_labels


# def create_label_report(gt_labels, model_labels):
#     """Map the model's labels to the neurons labels.
#     Parameters
#     ----------
#     gt_labels : ndarray
#         Label image with neurons labelled as mulitple values.
#     model_labels : ndarray
#         Label image from the model labelled as mulitple values.
#     Returns
#     -------
#     map_labels_existing: numpy array
#         The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label correctly labelled
#     map_fused_neurons: numpy array
#         The neurones are considered fused if they are labelled by the same model's label, in this case we will return The label value of the model and the label value of the neurone associated, the ratio of the pixels of the true label correctly labelled, the ratio of the pixels of the model's label that are in one of the fused neurones
#     new_labels: list
#         The labels of the model that are not labelled in the neurons, the ratio of the pixels of the model's label that are an artefact
#     """
#
#     map_labels_existing = []
#     map_fused_neurons = {}
#     "background_labels contains all model labels where gt_labels is 0 and model_labels is not 0"
#     background_labels = model_labels[np.where((gt_labels == 0))]
#     "new_labels contains all labels in model_labels for which more than PERCENT_CORRECT% of the pixels are not labelled in gt_labels"
#     new_labels = []
#     for lab in np.unique(background_labels):
#         if lab == 0:
#             continue
#         gt_background_size_at_lab = (
#             gt_labels[np.where((model_labels == lab) & (gt_labels == 0))]
#             .flatten()
#             .shape[0]
#         )
#         gt_lab_size = (
#             gt_labels[np.where(model_labels == lab)].flatten().shape[0]
#         )
#         if gt_background_size_at_lab / gt_lab_size > PERCENT_CORRECT:
#             new_labels.append(lab)
#
#     label_report_list = []
#     # label_report = {}  # contains a dict saying which labels are correct or wrong for each gt label
#     # model_label_values = {}  # contains the model labels value assigned to each unique gt label
#     not_found_id = 0
#
#     for i in tqdm(np.unique(gt_labels)):
#         if i == 0:
#             continue
#
#         gt_label = gt_labels[np.where(gt_labels == i)]  # get a single gt label
#
#         model_lab_on_gt = model_labels[
#             np.where(((gt_labels == i) & (model_labels != 0)))
#         ]  # all models labels on single gt_label
#         info = LabelInfo(i)
#
#         info.model_labels_id_and_status = {
#             label_id: "" for label_id in np.unique(model_lab_on_gt)
#         }
#
#         if model_lab_on_gt.shape[0] == 0:
#             info.model_labels_id_and_status[
#                 f"not_found_{not_found_id}"
#             ] = "not found"
#             not_found_id += 1
#             label_report_list.append(info)
#             continue
#
#         log.debug(f"model_lab_on_gt : {np.unique(model_lab_on_gt)}")
#
#         #  create LabelInfo object and init model_labels_id_and_status with all unique model labels on gt_label
#         log.debug(
#             f"info.model_labels_id_and_status : {info.model_labels_id_and_status}"
#         )
#
#         ratio = []
#         for model_lab_id in info.model_labels_id_and_status.keys():
#             size_model_label = (
#                 model_lab_on_gt[np.where(model_lab_on_gt == model_lab_id)]
#                 .flatten()
#                 .shape[0]
#             )
#             size_gt_label = gt_label.flatten().shape[0]
#
#             log.debug(f"size_model_label : {size_model_label}")
#             log.debug(f"size_gt_label : {size_gt_label}")
#
#             ratio.append(size_model_label / size_gt_label)
#
#         # log.debug(ratio)
#         ratio_model_lab_for_given_gt_lab = np.array(ratio)
#         info.best_model_label_coverage = ratio_model_lab_for_given_gt_lab.max()
#
#         best_model_lab_id = model_lab_on_gt[
#             np.argmax(ratio_model_lab_for_given_gt_lab)
#         ]
#         log.debug(f"best_model_lab_id : {best_model_lab_id}")
#
#         info.overall_gt_label_coverage = (
#             ratio_model_lab_for_given_gt_lab.sum()
#         )  # the ratio of the pixels of the true label correctly labelled
#
#         if info.best_model_label_coverage > PERCENT_CORRECT:
#             info.model_labels_id_and_status[best_model_lab_id] = "correct"
#             # info.model_labels_id_and_size[best_model_lab_id] = model_labels[np.where(model_labels == best_model_lab_id)].flatten().shape[0]
#         else:
#             info.model_labels_id_and_status[best_model_lab_id] = "wrong"
#         for model_lab_id in np.unique(model_lab_on_gt):
#             if model_lab_id != best_model_lab_id:
#                 log.debug(model_lab_id, "is wrong")
#                 info.model_labels_id_and_status[model_lab_id] = "wrong"
#
#         label_report_list.append(info)
#
#     correct_labels_id = []
#     for report in label_report_list:
#         for i_lab in report.model_labels_id_and_status.keys():
#             if report.model_labels_id_and_status[i_lab] == "correct":
#                 correct_labels_id.append(i_lab)
#     """Find all labels in label_report_list that are correct more than once"""
#     duplicated_labels = [
#         item for item, count in Counter(correct_labels_id).items() if count > 1
#     ]
#     "Sum up the size of all duplicated labels"
#     for i in duplicated_labels:
#         for report in label_report_list:
#             if (
#                 i in report.model_labels_id_and_status.keys()
#                 and report.model_labels_id_and_status[i] == "correct"
#             ):
#                 size = (
#                     model_labels[np.where(model_labels == i)]
#                     .flatten()
#                     .shape[0]
#                 )
#                 map_fused_neurons[i] = size
#
#     return label_report_list, new_labels, map_fused_neurons

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
