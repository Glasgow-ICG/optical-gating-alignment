"""Helper functions for sequence alignment algorithms.
Contains a function for resampling sequences and a function for accounting
for drift by taking the common window of two sequences."""

import numpy as np


def linear_interpolation(sequence, float_position, period=None, dtype=None):
    """A linear interpolation function for a 'sequence' of ND items
    Note: this is, currently, only for uint8 images. Why?
    """
    if dtype is None:
        dtype = sequence.dtype
    if period is None:
        period = len(sequence)

    # Bottom position
    # equivalent to np.floor(float_position).astype('int)
    lower_index = int(float_position)

    if float_position // 1 == float_position:
        # equivalent to sequence[np.floor(float_position).astype('int)]
        interpolated_value = (
            1.0 * sequence[lower_index]
        )  # 1.0* needed to force numba type
    else:

        # Interpolation Ratio
        interpolated_index = float_position - (float_position // 1)

        # Top position
        upper_index = lower_index + 1

        remainder = (float_position - lower_index) / float(upper_index - lower_index)

        # Values
        lower_value = sequence[lower_index]
        upper_value = sequence[int(upper_index % period)]

        interpolated_value = lower_value * (1 - remainder) + upper_value * remainder

    interpolated_value_int = interpolated_value.astype(dtype)

    return interpolated_value_int


def Interpolate(a, b, frac):
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

    # Original coordinates
    (_, m, n) = sequence.shape

    # Interpolated space coordinates
    p_indices_out = np.arange(0, period, 1 / interpolation_factor)  # supersample
    p_out = len(p_indices_out)

    # Sample at interpolated coordinates
    # # DEVNOTE: The boundary condition is dealt with simplistically
    # # ... but it works.
    # interpolated_sequence = np.zeros((p_out, m, n), dtype=dtype)
    # for i in np.arange(p_indices_out.shape[0]):
    #     if p_indices_out[i] + 1 > len(sequence):  # boundary condition
    #         interpolated_sequence[i, ...] = sequence[-1]
    #     else:
    #         interpolated_sequence[i, ...] = linear_interpolation(
    #             sequence, p_indices_out[i], period=period, dtype=dtype
    #         )

    # stolen from JT - see if this is better with boundaries
    result = []
    for i in np.arange(period * interpolation_factor):
        desiredPos = (i / float(period * interpolation_factor)) * period
        beforePos = int(desiredPos)
        afterPos = beforePos + 1
        before = sequence[beforePos]
        after = sequence[afterPos % len(sequence)]
        remainder = (desiredPos - beforePos) / float(afterPos - beforePos)

        image = Interpolate(before, after, remainder)
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
