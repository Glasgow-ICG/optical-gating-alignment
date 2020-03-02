"""Algorithm for phase matching two sequences based on cross-correlation."""

import numpy as np
from loguru import logger
import optical_gating_alignment.helper as hlp

# Set-up logger
logger.disable("optical-gating-alignment")


def cross_correlation(sequence1, sequence2):
    """Calculates cross correlation scores for two numpy arrays of order TXY"""
    temp = np.conj(np.fft.fft(sequence1, axis=0)) * np.fft.fft(sequence2, axis=0)
    temp2 = np.fft.ifft(temp, axis=0)

    scores = (
        np.sum(sequence1 * sequence1)
        + np.sum(sequence2 * sequence2)
        - 2 * np.sum(np.real(temp2), axis=1)
    )

    return scores


def v_minimum(y1, y2, y3):
    """Fit an even (symmetric) V to three points (yX) at x=-1, x=0 and x=+1"""
    if y1 > y3:
        x = 0.5 * (y1 - y3) / (y1 - y2)
        y = y2 - x * (y1 - y2)
    else:
        x = 0.5 * (y1 - y3) / (y3 - y2)
        y = y2 + x * (y3 - y2)

    return x, y


def minimum_score(scores):
    """Calculates the minimum position and value in a list of scores.
    Uses V-fitting for sub-integer accuracy."""
    # V-fitting for sub-integer interpolation
    # Note that scores is a ring vector
    # i.e. scores[0] is adjacent to scores[-1]

    y1 = scores[np.argmin(scores) - 1]  # Works even when minimum is at 0
    y2 = scores[np.argmin(scores)]
    y3 = scores[(np.argmin(scores) + 1) % len(scores)]
    minimum_position, minimum_value = v_minimum(y1, y2, y3)

    minimum_position = (minimum_position + np.argmin(scores)) % len(scores)

    return minimum_position, minimum_value


def rolling_cross_correlation(
    sequence1,
    sequence2,
    period1,
    period2,
    v_fitting=True,
    resampled_period=80,
    target=0,
):
    """Phase matching two sequences based on cross-correlation."""

    logger.debug("Original sequence #1:\t{0}", sequence1[:, 0, 0])
    logger.debug("Original sequence #2:\t{0}", sequence2[:, 0, 0])

    length1 = len(sequence1)
    length2 = len(sequence2)

    if period1 != resampled_period:
        sequence1 = hlp.resample_sequence(sequence1, period1, resampled_period)
    if period2 != resampled_period:
        sequence2 = hlp.resample_sequence(sequence2, period2, resampled_period)

    logger.debug("Resliced sequence #1:\t{0}", sequence1[:, 0, 0])
    logger.debug("Resliced sequence #2:\t{0}", sequence2[:, 0, 0])

    sequence1 = sequence1[:resampled_period].reshape([resampled_period, -1])
    sequence2 = sequence2[:resampled_period].reshape([resampled_period, -1])
    scores = cross_correlation(sequence1, sequence2)

    roll_factor, minimum_value = minimum_score(scores, v_fitting)
    roll_factor = (roll_factor / len(sequence1)) * length2

    alignment1 = (np.arange(0, length1) - roll_factor) % length1
    alignment2 = np.arange(0, length2)

    logger.info("Alignment 1:\t{0}", alignment1)
    logger.info("Alignment 2:\t{0}", alignment2)
    logger.info("Rolled by {0}", roll_factor)

    return (alignment1, alignment2, (target + roll_factor) % length2, minimum_value)


# if __name__ == "__main__":
#     logger.enable("optical-gating-alignment")
#     logger.info("Running toy example with")
#     string1 = [0, 1, 2, 3, 4, 3, 2, 2, 0]
#     string2 = [4, 3, 2, 1, 0, 0, 1, 1, 1, 2, 3, 4]
#     logger.info("Sequence #1: {0}", string1)
#     logger.info("Sequence #2: {0}", string2)
#     sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
#     sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
#     sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
#     sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
#     period1 = len(string1) - 1
#     period2 = len(string2) - 1

#     (alignment1, alignment2, roll_factor, score) = rolling_cross_correlation(
#         sequence1, sequence2, period1, period2
#     )

#     # Outputs for toy examples
#     string_out1 = []
#     string_out2 = []
#     for i in alignment1:
#         if i < 0:
#             string_out1.append(-1)
#         else:
#             string_out1.append(string1[int(round(i % period1))])
#     for i in alignment2:
#         if i < 0:
#             string_out2.append(-1)
#         else:
#             string_out2.append(string2[int(round(i % period2))])
#     logger.info("Aligned Sequence #1: {0}", string_out1)
#     logger.info("Aligned Sequence #2: {0}", string_out2)
