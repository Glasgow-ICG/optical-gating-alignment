"""Helper functions for sequence alignment algorithms.
Contains a function for resampling sequences and a function for accounting
for drift by taking the common window of two sequences."""

import numpy as np


def interpolate(a, b, frac):
    return a * (1 - frac) + b * frac


def interpolate_image_sequence(sequence, period, interpolation_factor=1, dtype=None):
    """Interpolate a series of images along a 'time' axis.
    Note: this is, currently, only for uint8 images. Why?

    Inputs:
    * series: a PxMxN numpy array contain P images of size MxN
      * P is a time-like axis, e.g. time or phase.
    * period: float period length in units of frames
    * interpolation_factor: integer interpolation factor, e.g. 2 doubles the series length

    Outputs:
    * interpolated_sequence: a P'xMxN numpy array
      * Contains np.ceil(interpolation_factor*period) frames, i.e. P' < =interpolation_factor*P
    """
    if dtype == None:
        dtype = sequence.dtype

    result = []
    for i in np.arange(period * interpolation_factor):
        desired_position = (i / float(period * interpolation_factor)) * period
        before_position = int(desired_position)
        after_position = before_position + 1
        before_value = sequence[before_position]
        after_value = sequence[after_position % len(sequence)]
        remainder_value = (desired_position - before_position) / float(
            after_position - before_position
        )

        image = interpolate(before_value, after_value, remainder_value)
        result.append(image)

    return np.array(result, dtype=dtype)


def drift_correction(sequence1, sequence2, drift):
    """Given a known x-y drift (drift), returns two sequences (sequence1 and
    sequence2) cropped by to their common window."""
    drift_x = drift[0]
    drift_y = drift[1]

    # Apply shifts [X1, X2, Y1, Y2]
    rectF = [0, sequence1[0].shape[0], 0, sequence1[0].shape[1]]
    rect = [0, sequence2[0].shape[0], 0, sequence2[0].shape[1]]

    if drift_x <= 0:
        rectF[0] = -drift_x
        rect[1] = rect[1] + drift_x
    else:
        rectF[1] = rectF[1] - drift_x
        rect[0] = drift_x
    if drift_y <= 0:
        rectF[2] = -drift_y
        rect[3] = rect[3] + drift_y
    else:
        rectF[3] = rectF[3] - drift_y
        rect[2] = +drift_y

    sequence1 = sequence1[:, rectF[0] : rectF[1], rectF[2] : rectF[3]]
    sequence2 = sequence2[:, rect[0] : rect[1], rect[2] : rect[3]]

    return sequence1, sequence2
