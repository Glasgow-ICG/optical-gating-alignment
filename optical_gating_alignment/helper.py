"""Helper functions for sequence alignment algorithms.
Contains a function for resampling sequences and a function for accounting
for drift by taking the common window of two sequences."""

import numpy as np


def linear_interpolation(sequence, float_position, period=None):
    """A linear interpolation function for a 'sequence' of ND items
    Note: this is, currently, only for uint8 images.
    """
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

    interpolated_value_int = interpolated_value.astype(sequence.dtype)

    return interpolated_value_int


def resample_sequence(sequence, current_period, new_period):
    """Resample a sequence from it's original length (current_period; float)
    to a new length (new_period; int)"""

    result = np.zeros(
        [new_period, sequence.shape[1], sequence.shape[2]], sequence.dtype
    )
    for i in range(new_period):
        resampled_position = (i / float(new_period)) * current_period
        result[i] = linear_interpolation(
            sequence, resampled_position, period=current_period
        )

    return result


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
