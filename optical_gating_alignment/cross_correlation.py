"""Algorithm for phase matching two sequences based on cross-correlation."""

import numpy as np
from loguru import logger
from . import helper as hlp
import j_py_sad_correlation as jps

# Set-up logger
logger.disable("optical_gating_alignment")


def cross_correlation(sequence1, sequence2, method="fft"):
    """Calculates cross correlation scores for each possible relative integer timeshifts between two numpy arrays (sequences) of order TXY.
    This uses the FFT for speed but is equivalent to the sum squared differences for each timeshift.
    Note: assumes both sequences are the same length."""

    # flatten to t x (x*y) array
    sequence1 = sequence1.reshape([len(sequence1), -1])
    sequence2 = sequence2.reshape([len(sequence2), -1])

    if method == "ssd":
        sequence1 = sequence1.astype("float", copy=False)
        sequence2 = sequence2.astype("float", copy=False)
        scores = []
        for roll in np.arange(len(sequence2)):
            sequence2_rolled = np.roll(sequence2, -roll, axis=0)
            SSD = 0
            for frame, frame_rolled in zip(sequence1, sequence2_rolled):
                SSD = SSD + np.sum((frame - frame_rolled) ** 2)
            scores.append(SSD)
        scores = np.array(scores)
    elif method == "fft" or method == None:  # default
        # The following is mathematically equivalent to the SSD but in Fourier space
        # It is generally faster than just the SSD (not for small cases though)
        sequence1 = sequence1.astype("float", copy=False)
        sequence2 = sequence2.astype("float", copy=False)
        temp = np.conj(np.fft.fft(sequence1, axis=0)) * np.fft.fft(sequence2, axis=0)
        temp2 = np.fft.ifft(temp, axis=0)
        # Compute the sum of squares of sequence1 and sequence2
        # (it is significantly faster to use np.dot compared to np.sum(a**2))
        # Note that these terms actually make no difference to where the minimum of the cross-correlation is,
        # but given that they're fast compared to the FFT, it keeps things consistent if we do calculate them.
        temp3 = np.dot(sequence1.ravel(), sequence1.ravel()) \
                + np.dot(sequence2.ravel(), sequence2.ravel())
        scores = temp3  - 2 * np.sum(np.real(temp2), axis=1)
    elif method == "jps":
        # The following is part of the C++ module py_sad_correlation
        # It is slower than the FFT but here for completeness
        scores = []
        for roll in np.arange(len(sequence2)):
            sequence2_rolled = np.roll(sequence2, -roll, axis=0)
            scores.append(jps.ssd_correlation(sequence1, sequence2_rolled)[0][0])
        scores = np.array(scores)

    return scores


def v_minimum(y1, y2, y3):
    """Fit an even (symmetric) V to three points (yX) at x=-1, x=0 and x=+1"""
    if y1 > y3:
        logger.debug("V-peak above y2")
        x = 0.5 * (y1 - y3) / (y1 - y2)
        y = y2 - x * (y1 - y2)
    else:
        logger.debug("V-peak below y2")
        x = 0.5 * (y1 - y3) / (y3 - y2)
        y = y2 + x * (y3 - y2)

    return x, y


def minimum_score(scores):
    """Calculates the minimum position and value in a list of scores.
    Uses V-fitting for sub-integer accuracy."""
    # V-fitting for sub-integer interpolation
    # Note that scores is a ring vector
    # i.e. scores[0] is adjacent to scores[-1]

    xmin = np.argmin(scores)
    y1 = scores[xmin - 1]  # Works even when minimum is at 0
    y2 = scores[xmin]
    y3 = scores[(xmin + 1) % len(scores)]
    logger.info("{0}, {1}, {2}", y1, y2, y3)
    minimum_position, minimum_value = v_minimum(y1, y2, y3)

    minimum_position = (minimum_position + xmin) % len(scores)

    logger.info(
        "Argmin: {0} ({1}); interpolated min: {2} ({3});",
        xmin,
        y2,
        minimum_position,
        minimum_value,
    )
    if minimum_value < 0:
        logger.warning("Interpolated minimum is below zero! Setting to 0.1.")
        minimum_value = 0.1
        # FIXME

    return minimum_position, minimum_value


def rolling_cross_correlation(
    sequence1, sequence2, period1, period2, resampled_period=80, method="fft"
):
    """Phase matching two sequences based on cross-correlation.
    Note: resampled_period=None only to be used when the user has made sure both sequences are the same number of frames long."""

    logger.debug(
        "Original sequence #1:\t{0};\t{1};", sequence1[:, 0, 0], sequence1.shape
    )
    logger.debug(
        "Original sequence #2:\t{0};\t{1};", sequence2[:, 0, 0], sequence2.shape
    )

    length1 = len(sequence1)
    length2 = len(sequence2)

    if resampled_period is not None and period1 != resampled_period:
        sequence1 = hlp.interpolate_image_sequence(
            sequence1, period1, resampled_period / period1
        )[:resampled_period]
        sequence1 = sequence1[:resampled_period].reshape([resampled_period, -1])
    else:
        sequence1 = sequence1.reshape([length1, -1])
    if resampled_period is not None and period2 != resampled_period:
        sequence2 = hlp.interpolate_image_sequence(
            sequence2, period2, resampled_period / period2
        )[:resampled_period]
        sequence2 = sequence2[:resampled_period].reshape([resampled_period, -1])
    else:
        sequence2 = sequence2.reshape([length2, -1])
    logger.debug(
        "Resliced and reshaped sequence #1:\t{0};\t{1};",
        sequence1[:, 0],
        sequence1.shape,
    )
    logger.debug(
        "Resliced and reshaped sequence #2:\t{0};\t{1};",
        sequence2[:, 0],
        sequence2.shape,
    )

    scores = cross_correlation(sequence1, sequence2, method=method)
    logger.debug("Scores: {0}", scores)

    roll_factor, minimum_value = minimum_score(scores)
    logger.debug("Scores suggest global roll of {0} ({1}).", roll_factor, minimum_value)
    if resampled_period is not None:
        roll_factor = (roll_factor / resampled_period) * length2

    alignment1 = (np.arange(0, length1) - roll_factor) % length1
    alignment2 = np.arange(0, length2)

    logger.debug("Alignment 1:\t{0}", alignment1)
    logger.debug("Alignment 2:\t{0}", alignment2)
    logger.info("Rolled by {0}", roll_factor)
    logger.debug("Score: {0}", minimum_value)

    return roll_factor % length2, minimum_value
