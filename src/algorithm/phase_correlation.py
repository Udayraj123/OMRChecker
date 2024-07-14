import cv2
import numpy as np


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

    corr_image = cv2.normalize(corr, 0, 255, norm_type=cv2.NORM_MINMAX)

    return [int(x), int(y)], corr_image
