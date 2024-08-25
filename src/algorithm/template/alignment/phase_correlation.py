import cv2
import numpy as np

from src.algorithm.template.alignment.utils import show_displacement_overlay
from src.utils.interaction import InteractionUtils
from src.utils.logger import logger


def phase_correlation(a, b):
    R = np.fft.fft2(a) * np.fft.fft2(b).conj()
    r = np.abs(np.fft.ifft2(R))
    return r


def get_phase_correlation_shifts(alignment_image, gray_image):
    corr = phase_correlation(alignment_image, gray_image)

    shape = corr.shape
    maxima = np.unravel_index(np.argmax(corr), shape)

    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    x, y = shifts[:2]

    corr_image = cv2.normalize(corr, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    return [int(x), int(y)], corr_image


def apply_phase_correlation_shifts(
    field_block, block_gray_alignment_image, block_gray_image
):
    field_block.shifts, corr_image = get_phase_correlation_shifts(
        block_gray_alignment_image, block_gray_image
    )
    logger.info(field_block.name, field_block.shifts)

    M = np.float32(
        [[1, 0, -1 * field_block.shifts[0]], [0, 1, -1 * field_block.shifts[1]]]
    )
    shifted_block_image = cv2.warpAffine(
        block_gray_image, M, (block_gray_image.shape[1], block_gray_image.shape[0])
    )
    InteractionUtils.show("Correlation", corr_image, 0)

    show_displacement_overlay(
        block_gray_alignment_image, block_gray_image, shifted_block_image
    )

    return shifted_block_image
