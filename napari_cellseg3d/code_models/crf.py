"""Implements the CRF post-processing step for the W-Net.

The CRF requires the following parameters:

* images : Array of shape (N, C, H, W, D) containing the input images.
* predictions: Array of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
* sa: alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
* sb: beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
* sg: gamma standard deviation, the scale of the smoothness/gaussian kernel.
* w1: weight of the appearance/bilateral kernel.
* w2: weight of the smoothness/gaussian kernel.

Inspired by https://arxiv.org/abs/1606.00915 and https://arxiv.org/abs/1711.08506.
Also uses research from:
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
Philipp Krähenbühl and Vladlen Koltun
NIPS 2011

Implemented using the pydense library available at https://github.com/lucasb-eyer/pydensecrf.
"""
import importlib

import numpy as np
from napari.qt.threading import GeneratorWorker

from napari_cellseg3d.config import CRFConfig
from napari_cellseg3d.utils import LOGGER as logger

spec = importlib.util.find_spec("pydensecrf")
CRF_INSTALLED = spec is not None
if not CRF_INSTALLED:
    logger.info(
        "pydensecrf not installed, CRF post-processing will not be available. "
        "Please install by running : pip install pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git#egg=master"
        "This is not a hard requirement, you do not need it to install it unless you want to use the CRF post-processing step."
    )
else:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        create_pairwise_bilateral,
        create_pairwise_gaussian,
        unary_from_softmax,
    )

__author__ = "Yves Paychère, Colin Hofmann, Cyril Achard"
__credits__ = [
    "Yves Paychère",
    "Colin Hofmann",
    "Cyril Achard",
    "Philipp Krähenbühl",
    "Vladlen Koltun",
    "Liang-Chieh Chen",
    "George Papandreou",
    "Iasonas Kokkinos",
    "Kevin Murphy",
    "Alan L. Yuille",
    "Xide Xia",
    "Brian Kulis",
    "Lucas Beyer",
]


def correct_shape_for_crf(image, desired_dims=4):
    """Corrects the shape of the image to be compatible with the CRF post-processing step."""
    logger.debug(f"Correcting shape for CRF, desired_dims={desired_dims}")
    logger.debug(f"Image shape: {image.shape}")
    if len(image.shape) > desired_dims:
        # if image.shape[0] > 1:
        #     raise ValueError(
        #         f"Image shape {image.shape} might have several channels"
        #     )
        image = np.squeeze(image, axis=0)
    elif len(image.shape) < desired_dims:
        image = np.expand_dims(image, axis=0)
    logger.debug(f"Corrected image shape: {image.shape}")
    return image


def crf_batch(images, probs, sa, sb, sg, w1, w2, n_iter=5):
    """CRF post-processing step for the W-Net, applied to a batch of images.

    Args:
        images (np.ndarray): Array of shape (N, C, H, W, D) containing the input images.
        probs (np.ndarray): Array of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
        sa (float): alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
        sb (float): beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
        sg (float): gamma standard deviation, the scale of the smoothness/gaussian kernel.
        w1 (float): weight of the appearance/bilateral kernel.
        w2 (float): weight of the smoothness/gaussian kernel.
        n_iter (int, optional): Number of iterations for the CRF post-processing step. Defaults to 5.

    Returns:
        np.ndarray: Array of shape (N, K, H, W, D) containing the refined class probabilities for each pixel.
    """
    if not CRF_INSTALLED:
        return None

    return np.stack(
        [
            crf(images[i], probs[i], sa, sb, sg, w1, w2, n_iter=n_iter)
            for i in range(images.shape[0])
        ],
        axis=0,
    )


def crf(image, prob, sa, sb, sg, w1, w2, n_iter=5):
    """Implements the CRF post-processing step for the W-Net.

    Inspired by https://arxiv.org/abs/1210.5644, https://arxiv.org/abs/1606.00915 and https://arxiv.org/abs/1711.08506.
    Implemented using the pydensecrf library.

    Args:
        image (np.ndarray): Array of shape (C, H, W, D) containing the input image.
        prob (np.ndarray): Array of shape (K, H, W, D) containing the predicted class probabilities for each pixel.
        sa (float): alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
        sb (float): beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
        sg (float): gamma standard deviation, the scale of the smoothness/gaussian kernel.
        w1 (float): weight of the appearance/bilateral kernel.
        w2 (float): weight of the smoothness/gaussian kernel.
        n_iter (int, optional): Number of iterations for the CRF post-processing step. Defaults to 5.

    Returns:
        np.ndarray: Array of shape (K, H, W, D) containing the refined class probabilities for each pixel.
    """
    if not CRF_INSTALLED:
        return None

    d = dcrf.DenseCRF(
        image.shape[1] * image.shape[2] * image.shape[3], prob.shape[0]
    )
    # print(f"Image shape : {image.shape}")
    # print(f"Prob shape : {prob.shape}")
    # d = dcrf.DenseCRF(262144, 3) # npoints, nlabels

    # Get unary potentials from softmax probabilities
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)

    # Generate pairwise potentials
    featsGaussian = create_pairwise_gaussian(
        sdims=(sg, sg, sg), shape=image.shape[1:]
    )  # image.shape)
    featsBilateral = create_pairwise_bilateral(
        sdims=(sa, sa, sa),
        schan=tuple([sb for i in range(image.shape[0])]),
        img=image,
        chdim=-1,
    )

    # Add pairwise potentials to the CRF
    compat = np.ones(prob.shape[0], dtype=np.float32) - np.diag(
        [1 for i in range(prob.shape[0])]
        # , dtype=np.float32
    )
    d.addPairwiseEnergy(featsGaussian, compat=compat.astype(np.float32) * w2)
    d.addPairwiseEnergy(featsBilateral, compat=compat.astype(np.float32) * w1)

    # Run inference
    Q = d.inference(n_iter)

    return np.array(Q).reshape(
        (prob.shape[0], image.shape[1], image.shape[2], image.shape[3])
    )


def crf_with_config(image, prob, config: CRFConfig = None, log=logger.info):
    """Implements the CRF post-processing step for the W-Net.

    Args:
        image (np.ndarray): Array of shape (C, H, W, D) containing the input image.
        prob (np.ndarray): Array of shape (K, H, W, D) containing the predicted class probabilities for each pixel.
        config (CRFConfig, optional): Configuration for the CRF post-processing step. Defaults to None.
        log (function, optional): Logging function. Defaults to logger.info.
    """
    if config is None:
        config = CRFConfig()
    if image.shape[-3:] != prob.shape[-3:]:
        raise ValueError(
            f"Image and probability shapes do not match: {image.shape} vs {prob.shape}"
            f" (expected {image.shape[-3:]} == {prob.shape[-3:]})"
        )

    image = correct_shape_for_crf(image)
    prob = correct_shape_for_crf(prob)

    if log is not None:
        log("Running CRF post-processing step")
        log(f"Image shape : {image.shape}")
        log(f"Labels shape : {prob.shape}")

    return crf(
        image,
        prob,
        config.sa,
        config.sb,
        config.sg,
        config.w1,
        config.w2,
        config.n_iters,
    )


class CRFWorker(GeneratorWorker):
    """Worker for the CRF post-processing step for the W-Net."""

    def __init__(
        self,
        images_list: list,
        labels_list: list,
        config: CRFConfig = None,
        log=None,
    ):
        """Initializes the CRFWorker.

        Args:
            images_list (list): List of images to process.
            labels_list (list): List of labels to process.
            config (CRFConfig, optional): Configuration for the CRF post-processing step. Defaults to None.
            log (function, optional): Logging function. Defaults to None.
        """
        super().__init__(self._run_crf_job)

        self.images = images_list
        self.labels = labels_list
        if config is None:
            self.config = CRFConfig()
        else:
            self.config = config
        self.log = log

    def _run_crf_job(self):
        """Runs the CRF post-processing step for the W-Net."""
        if not CRF_INSTALLED:
            raise ImportError("pydensecrf is not installed.")

        if len(self.images) != len(self.labels):
            raise ValueError("Number of images and labels must be the same.")

        for i in range(len(self.images)):
            if self.images[i].shape[-3:] != self.labels[i].shape[-3:]:
                raise ValueError("Image and labels must have the same shape.")

            im = correct_shape_for_crf(self.images[i])
            prob = correct_shape_for_crf(self.labels[i])

            logger.debug(f"image shape : {im.shape}")
            logger.debug(f"labels shape : {prob.shape}")

            yield crf(
                im,
                prob,
                self.config.sa,
                self.config.sb,
                self.config.sg,
                self.config.w1,
                self.config.w2,
                n_iter=self.config.n_iters,
            )
