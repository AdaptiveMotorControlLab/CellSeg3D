"""
Implements the CRF post-processing step for the W-Net.
Inspired by https://arxiv.org/abs/1606.00915 and https://arxiv.org/abs/1711.08506.

Also uses research from:
Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
Philipp Krähenbühl and Vladlen Koltun
NIPS 2011

Implemented using the pydense libary available at https://github.com/lucasb-eyer/pydensecrf.
"""

from warnings import warn

import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral
    from pydensecrf.utils import create_pairwise_gaussian
    from pydensecrf.utils import unary_from_softmax

    CRF_INSTALLED = True
except ImportError:
    warn(
        "pydensecrf not installed, CRF post-processing will not be available. "
        "Please install by running pip install cellseg3d[crf]"
    )
    CRF_INSTALLED = False

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


def crf_batch(images, probs, sa, sb, sg, w1, w2, n_iter=5):
    """CRF post-processing step for the W-Net, applied to a batch of images.

    Args:
        images (np.ndarray): Array of shape (N, C, H, W, D) containing the input images.
        probs (np.ndarray): Array of shape (N, K, H, W, D) containing the predicted class probabilities for each pixel.
        sa (float): alpha standard deviation, the scale of the spatial part of the appearance/bilateral kernel.
        sb (float): beta standard deviation, the scale of the color part of the appearance/bilateral kernel.
        sg (float): gamma standard deviation, the scale of the smoothness/gaussian kernel.

    Returns:
        np.ndarray: Array of shape (N, K, H, W, D) containing the refined class probabilities for each pixel.
    """

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

    Returns:
        np.ndarray: Array of shape (K, H, W, D) containing the refined class probabilities for each pixel.
    """
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
