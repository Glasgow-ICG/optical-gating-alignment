"""Modules for phase matching two sequences based on sequence alignment.
Uses a cascading form of the Needleman Wunsch algorithm.
This module includes all necessary functions."""

import optical_gating_alignment.helper as hlp
import numpy as np
from loguru import logger
import j_py_sad_correlation as jps

# Set-up logger
logger.disable("optical-gating-alignment")


def get_roll_factor_at(alignment1, alignment2, phase1):
    """Get the precise roll factor for a known phase (phase1) in one sequence based on two alignments (alignment1 and alignment2). This allows the system to deal with arrhythmic sequences and the consequent indels."""

    if isinstance(alignment1, list):
        alignment1 = np.array(alignment1)

    length1 = len(alignment1)

    # First get the exact index of phase1 in alignment1
    # Case 1: where phase1 is in alignment1
    # i.e. phase1 likely to be a whole number
    if phase1 in alignment1:
        # DEVNOTE: we assume there is only ever one, which should be true or something has gone awry
        idxPos = np.nonzero(alignment1 == phase1)[0][0]
        logger.info("Exact phase found at index {0} in alignment 1", idxPos)
    # Case 2: where phase1 is not in alignment1
    # find the indices in alignment1 that are the lower and upper bounds
    # then use linear interpolation to get the exact index
    else:
        # set starting bounds to a non-index
        lower_bound = -1
        upper_bound = -1
        # DEVNOTE: the allow flag deals with the scenario that my alignment starts after the desired phase, i.e. alignment[0]>phase1, and wraps latter in the sequence
        allow = False  # only turn on if I've seen a value smaller than desired
        for idx1 in range(length1):
            logger.debug("{0} {1}", idx1, alignment1[idx1])
            logger.debug(
                "{0} {1} {2} {3}",
                alignment1[idx1] >= 0,
                allow,
                alignment1[idx1] == min(alignment1[alignment1 > 0]),
                alignment1[idx1] > phase1,
            )
            if alignment1[idx1] >= 0 and not allow and alignment1[idx1] < phase1:
                # not gap (-1) and not started searching and lower than phase1
                allow = True  # start searching for bounds
                logger.debug(
                    "Began searching for bounds at alignment1[{0}]={1}.",
                    idx1,
                    alignment1[idx1],
                )
            elif (
                alignment1[idx1] >= 0
                and not allow
                and alignment1[idx1] == min(alignment1[alignment1 > 0])
                and alignment1[idx1] > phase1
            ):
                # not gap and not started and is smallest in alignment1
                # but is greater than phase, i.e. target is between the wrap point
                logger.info("Desired phase at alignment sequence 1 wrap point (type1)")
                # set lower and upper bound and stop searching
                lower_bound = idx1 - 1
                upper_bound = idx1
                logger.debug(
                    "Preliminary bounds applied: {0}, {1} for {2}",
                    lower_bound,
                    upper_bound,
                    phase1,
                )
                break
            elif allow and alignment1[idx1] >= phase1:
                # started and higher than phase1 (not gap implied)
                lower_bound = idx1 - 1
                upper_bound = idx1
                logger.debug(
                    "Preliminary bounds applied: {0}, {1} for {2}",
                    lower_bound,
                    upper_bound,
                    phase1,
                )
                break
            elif allow and alignment1[idx1] == 0:
                # started and wrap point (not gap implied)
                logger.info("Desired phase at alignment sequence 1 wrap point (type2)")
                lower_bound = idx1 - 1
                upper_bound = idx1
                logger.debug(
                    "Preliminary bounds applied: {0}, {1} for {2}",
                    lower_bound,
                    upper_bound,
                    phase1,
                )
                break

        if lower_bound < 0 and upper_bound < 0 and not allow:
            # no valid bounds set
            # assume phase1 completely not in alignment then
            logger.critical("Phase not found in alignment sequence 1")
            logger.debug(phase1, alignment1)
            return None

        # assume lower_bound < 0 and upper_bound < 0 and allow
        # started searching but didn't set bounds
        # assume bounds are around the wrap point
        logger.info("Wrapping around alignment sequence 1")
        lower_bound = idx1
        upper_bound = idx1 + 1
        while alignment1[upper_bound % length1] < 0:
            upper_bound = upper_bound + 1
        logger.debug(
            "Preliminary bounds applied: {0}, {1} for {2}",
            lower_bound,
            upper_bound,
            phase1,
        )

        # DEVNOTE: currently the lower and upper bounds don't deal with any gaps, i.e. -1s
        # account for gaps in lower bound
        while alignment1[lower_bound] < 0:
            # if currently gap keep lowering the lower bound
            lower_bound = lower_bound - 1
        logger.info(
            "Interpolating with lower bound of {0} and upper bound of {1}",
            lower_bound,
            upper_bound % length1,
        )

        interpolated_index1 = (phase1 - alignment1[lower_bound]) / (
            alignment1[upper_bound % length1] - alignment1[lower_bound]
        )

        logger.info("Phase positioned at interpolated index {0} in alignment 1", idxPos)
        idxPos = (interpolated_index1 * (upper_bound - lower_bound)) + lower_bound

        logger.debug(
            phase1,
            lower_bound,
            alignment1[lower_bound],
            upper_bound,
            length1,
            upper_bound % length1,
            alignment1[upper_bound % length1],
        )
        logger.debug(interpolated_index1)

    # Map precise index to alignment2, considering gaps
    # case where phase captured in alignment1 is integer and valid in alignment2
    phase2 = None

    # DEVNOTE: the middle criterion is needed because alignments may not be the same length
    if (
        (idxPos // 1) == idxPos
        and idxPos < len(alignment2)
        and alignment2[int(idxPos)] >= 0
    ):
        # index is integer, less than the length of alignment sequence 2 and that position isn't a gap
        phase2 = alignment2[int(idxPos)]
        logger.info("Exact index used in alignment 2 to give a phase of {0}", phase1)
        logger.debug(alignment2[int(idxPos)])

    else:
        length2 = len(alignment2)
        alignment2_lower_bound = int(idxPos)  # same as np.floor
        alignment2_upper_bound = int(idxPos + 1)  # same as np.ceil
        logger.debug("{0} {1}", alignment2_lower_bound, alignment2_upper_bound)
        logger.debug("{0} {1}", length1, length2)

        # check not same value (occurs when exactly hits an index)
        if alignment2_lower_bound == alignment2_upper_bound:
            alignment2_upper_bound = alignment2_upper_bound + 1

        # deal with gaps in alignment2
        while alignment2[int(alignment2_lower_bound % length2)] < 0:
            alignment2_lower_bound = alignment2_lower_bound - 1
        while alignment2[int(alignment2_upper_bound % length2)] < 0:
            alignment2_upper_bound = alignment2_upper_bound + 1

        # interpolate for exact position
        logger.debug(
            "Interpolating with lower bound of {0} and upper bound of {1}",
            alignment2_lower_bound,
            alignment2_upper_bound % length2,
        )

        interpolated_index2 = (idxPos - alignment2_lower_bound) / (
            alignment2_upper_bound - alignment2_lower_bound
        )
        if (
            alignment2[alignment2_upper_bound % length2]
            < alignment2[alignment2_lower_bound]
        ):
            phase2 = (
                interpolated_index2
                * (
                    (
                        alignment2[alignment2_lower_bound]
                        + alignment2[alignment2_upper_bound % length2]
                        + 1
                    )
                    - alignment2[alignment2_lower_bound]
                )
            ) + alignment2[alignment2_lower_bound]
        else:
            phase2 = (
                interpolated_index2
                * (
                    alignment2[alignment2_upper_bound % length2]
                    - alignment2[alignment2_lower_bound]
                )
            ) + alignment2[alignment2_lower_bound]
        logger.info(
            "Interpolated index used to calculate phase of {0} in alignment 2", phase2
        )

        phase2 = phase2 % length2

    if phase2 is None:
        logger.critical("No phase calculated for alignment sequence 2")
    return phase2


def interpolate_image_sequence(sequence, period, interpolation_factor=1):
    """Interpolate a series of images along a 'time' axis.
    Note: this is, currently, only for uint8 images

    Inputs:
    * series: a PxMxN numpy array contain P images of size MxN
      * P is a time-like axis, e.g. time or phase.
    * period: float period length in units of frames
    * interpolation_factor: integer interpolation factor, e.g. 2 doubles the series length

    Outputs:
    * interpolated_sequence: a P'xMxN numpy array
      * Contains np.ceil(interpolation_factor*period) frames, i.e. P' < =interpolation_factor*P
    """

    # Original coordinates
    (_, m, n) = sequence.shape

    # Interpolated space coordinates
    p_indices_out = np.arange(0, period, 1 / interpolation_factor)  # supersample
    p_out = len(p_indices_out)

    # Sample at interpolated coordinates
    # DEVNOTE: The boundary condition is dealt with simplistically
    # ... but it works.
    interpolated_sequence = np.zeros((p_out, m, n), dtype=np.uint8)
    for i in np.arange(p_indices_out.shape[0]):
        if p_indices_out[i] + 1 > len(sequence):  # boundary condition
            interpolated_sequence[i, ...] = sequence[-1]
        else:
            interpolated_sequence[i, ...] = hlp.linear_interpolation(
                sequence, p_indices_out[i]
            )

    return interpolated_sequence


def fill_traceback_matrix(score_matrix, gap_penalty=0):
    """Using a score matrix, fill out a traceback matrix.
    This can then be traversed to identify a valid alignment.
    """
    traceback_matrix = np.zeros(
        (score_matrix.shape[0] + 1, score_matrix.shape[1] + 1), dtype=np.float64
    )

    for t2 in np.arange(traceback_matrix.shape[0]):  # for all rows

        if t2 == 0:  # if the first row
            for t1 in np.arange(
                1, traceback_matrix.shape[1]
            ):  # for all but first column

                match_score = score_matrix[
                    t2 - 1, t1 - 1
                ]  # get score for this combination (i.e. high score for a match)

                # left == insert gap into sequenceA
                insert = (
                    traceback_matrix[t2, t1 - 1] - gap_penalty - match_score
                )  # get score to the left plus the gap_penalty (same as (t1)*gap_penalty)

                traceback_matrix[t2, t1] = insert

        else:  # if any but the first row

            for t1 in np.arange(traceback_matrix.shape[1]):  # for all columns
                if t1 == 0:  # if the first column

                    match_score = score_matrix[
                        t2 - 1, t1 - 1
                    ]  # get score for this combination (i.e. high score for a match)

                    # above == insert gap into sequenceB (or delete frame for sequenceA)
                    delete = (
                        traceback_matrix[t2 - 1, t1] - gap_penalty - match_score
                    )  # get score to the above plus the gap_penalty (same as t2*gap_penalty)

                    traceback_matrix[t2, t1] = delete  # - match_score

                else:

                    match_score = score_matrix[
                        t2 - 1, t1 - 1
                    ]  # get score for this combination (i.e. high score for a match)

                    # diagonal
                    match = traceback_matrix[t2 - 1, t1 - 1] - match_score

                    # above
                    delete = traceback_matrix[t2 - 1, t1] - gap_penalty - match_score

                    # left
                    insert = traceback_matrix[t2, t1 - 1] - gap_penalty - match_score

                    traceback_matrix[t2, t1] = max(
                        [match, insert, delete]
                    )  # get maximum score from left, left above and above

    return traceback_matrix


def roll_score_matrix(score_matrix, roll_factor=0, axis=0):
    """Utility function to roll a 2D matrix along a given axis."""
    rolled_score_matrix = np.zeros(score_matrix.shape, dtype=score_matrix.dtype)
    for i in np.arange(score_matrix.shape[axis]):
        if axis == 0:
            rolled_score_matrix[i, :] = score_matrix[
                (i - roll_factor) % score_matrix.shape[0], :
            ]
        elif axis == 1:
            rolled_score_matrix[:, i] = score_matrix[
                :, (i - roll_factor) % score_matrix.shape[1]
            ]
    return rolled_score_matrix


def construct_cascade(score_matrix, gap_penalty=0, axis=0):
    """Create a 'cascade' of score arrays for use in the Needleman-Wunsch algorith.

    Inputs:
    * score_matrix: a score MxN array between two semi-periodic sequences
      * Columns represent one sequence of length M; rows the another of length N
    * gap_penalty: the Needleman-Wunsch penalty for introducing a gap (zero means no penalty, large means big penalty, i.e. less likely).
    * axis: the axis along which to roll/cascade

    Outputs:
    * cascades: a MxNx[M/N] array of cascaded traceback matrices
      * The third dimension depends on the axis parameter
    """

    # Create 3D array to hold all cascades
    cascades = np.zeros(
        (score_matrix.shape[0] + 1, score_matrix.shape[1] + 1, score_matrix.shape[0]),
        dtype=np.float64,
    )

    # Create a new cascaded score array for each alignment (by rolling along axis)
    for n in np.arange(
        score_matrix.shape[1 - axis]
    ):  # the 1-axis tricks means we loop over 0 if axis=1 and vice versa
        logger.info("Getting score matrix for roll of {0} frames...", n)
        cascades[:, :, n] = fill_traceback_matrix(score_matrix, gap_penalty=gap_penalty)
        score_matrix = roll_score_matrix(score_matrix, 1, axis=axis)

    return cascades


def traverse_traceback_matrix(sequence, template_sequence, traceback_matrix):
    """Traverse a tracbeack matrix and return aligned versions of two sequences.
    The tempkate_sequence is less likely to have indels inserted."""
    x = template_sequence.shape[0]
    y = sequence.shape[0]

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignmentA = []
    alignmentB = []
    while traversing:
        options = np.zeros((3,))

        x_up = x - 1
        y_left = y - 1
        logger.debug("-----")
        logger.debug(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "curr",
            x,
            y,
            traceback_matrix[x, y],
            sequence[-y, 0, 0],
            template_sequence[-x, 0, 0],
        )
        logger.debug(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "diag",
            x_up,
            y_left,
            traceback_matrix[x_up, y_left],
            sequence[-y_left, 0, 0],
            template_sequence[-x_up, 0, 0],
        )
        logger.debug(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "up  ",
            x_up,
            y,
            traceback_matrix[x_up, y],
            sequence[-y, 0, 0],
            template_sequence[-x_up, 0, 0],
        )
        logger.debug(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "left",
            x,
            y_left,
            traceback_matrix[x, y_left],
            sequence[-y_left, 0, 0],
            template_sequence[-x, 0, 0],
        )
        if x_up >= 0:
            if y_left >= 0:
                options[:] = [
                    traceback_matrix[x_up, y_left],
                    traceback_matrix[x_up, y],
                    traceback_matrix[x, y_left],
                ]
            else:
                logger.info("Boundary Condition:\tI'm at the left")
                options[:] = [-np.inf, traceback_matrix[x_up, y], -np.inf]
        else:
            logger.info("Boundary Condition:\tI'm at the top")
            if y_left >= 0:
                options[:] = [-np.inf, -np.inf, traceback_matrix[x, y_left]]
            else:
                logger.warning("Boundary Condition:\tI'm at the top left")
                logger.warning("Boundary Condition:\tI should not have got here!")
                break
        direction = np.argmax(options)

        if direction == 1:
            alignmentA.append(-1)
            alignmentB.append(x_up)
            x = x_up
            logger.info("Direction Travelled:\tI've gone up")
        elif direction == 0:
            alignmentA.append(y_left)
            alignmentB.append(x_up)
            x = x_up
            y = y_left
            logger.info("Direction Travelled:\tI've gone diagonal")
        elif direction == 2:
            alignmentA.append(y_left)
            alignmentB.append(-1)
            y = y_left
            logger.info("Direction Travelled:\tI've gone left")
        if x == 0 and y == 0:
            logger.info("Traversing Complete")
            traversing = False

    # Reverses sequence
    alignmentA = np.asarray(alignmentA[::-1], dtype=np.float)
    alignmentB = np.asarray(alignmentB[::-1], dtype=np.float)

    return alignmentA, alignmentB


def cascading_needleman_wunsch(
    sequence,
    template_sequence,
    period,
    template_period,
    gap_penalty=0,
    interpolation_factor=None,
    ref_seq_phase=0,
):
    """Calculating the cascading Needleman-Wunsch alignment for two semi-periodic sequences.

    For the two sequences provided, this algorithm will assume the second is the 'template'.
    The template sequence will see only minor changes for alignment, i.e. adding gaps.
    The other sequence will see rolling and gap addition.

    Inputs:
    * sequence, template_sequence: a PxMxN numpy array representing the two periods to align
    * period, remplatePeriod: the float period for sequence/template_sequence in frame units (caller must determine this)
    * gap_penalty: the Needleman-Wunsch penalty for introducing a gap as a percentage (relating to the calculated score matrix)
    * interpolation_factor: integer linear interpolation factor, e.g. a factor of 2 will double the image resolution along P
    * ref_seq_phase: integer frame (in B) for which to return the roll factor
    """
    if template_period is None:
        template_period = template_sequence.shape[0]
    if period is None:
        period = sequence.shape[0]

    logger.info(
        "Sequence #1 has {0} frames and sequence #2 has {1} frames",
        len(sequence),
        len(template_sequence),
    )

    if interpolation_factor is not None:
        logger.info(
            "Interpolating by a factor of {0} for greater precision...",
            interpolation_factor,
        )
        sequence = interpolate_image_sequence(
            sequence, period, interpolation_factor=interpolation_factor
        )
        template_sequence = interpolate_image_sequence(
            template_sequence, period, interpolation_factor=interpolation_factor
        )
        logger.info(
            "\tSequence #1 now has {0} frames and sequence #2 now has {1} frames:",
            len(sequence),
            len(template_sequence),
        )

    # Calculate Score Matrix - C++
    score_matrix = jps.sad_grid(sequence, template_sequence)

    logger.debug("Score Matrix:")
    logger.debug(score_matrix)
    logger.debug(
        "\tDtype: {0};\tShape: ({1},{2})",
        score_matrix.dtype,
        score_matrix.shape[0],
        score_matrix.shape[1],
    )

    # Cascade the SAD Grid
    cascades = construct_cascade(score_matrix, gap_penalty=gap_penalty, axis=1)
    logger.debug("Unrolled Traceback Matrix:")
    logger.debug(cascades[:, :, 0])
    logger.debug(
        "\tDtype: {0};\tShape: ({1},{2},{3})",
        cascades.dtype,
        cascades.shape[0],
        cascades.shape[1],
        cascades.shape[2],
    )

    # Pick Cascade and Roll sequence
    roll_factor = np.argmax(cascades[-1, -1, :])
    score = cascades[-1, -1, roll_factor]
    score = (score + (np.iinfo(sequence.dtype).max * sequence.size / 10)) / (
        np.iinfo(sequence.dtype).max * sequence.size / 10
    )
    if score <= 0:
        logger.warning("Negative Score")
    score = 0 if score < 0 else score

    traceback_matrix = cascades[:, :, roll_factor]
    sequence = np.roll(sequence, roll_factor, axis=2)
    logger.info("Chose cascade {0} of {1}:", roll_factor, len(sequence))
    logger.debug("Cascaded traceback matrixes:")
    logger.debug(traceback_matrix)
    logger.debug(
        "Cascade scores:\t", cascades[len(template_sequence), len(sequence), :]
    )
    logger.debug(
        "Shape: ({0},{1})", traceback_matrix.shape[0], traceback_matrix.shape[1]
    )

    (alignmentAWrapped, alignmentB) = traverse_traceback_matrix(
        sequence, template_sequence, traceback_matrix
    )

    logger.info("roll_factor (interpolated):\t{0}", roll_factor)
    logger.info("Aligned sequence #1 (interpolated, wrapped):\t{0}", alignmentAWrapped)
    logger.info("Aligned sequence #2 (interpolated):\t\t\t{0}", alignmentB)

    if interpolation_factor is not None:
        logger.info("De-interpolating for result...")
        # Divide by interpolation factor and modulo period
        # ignore -1s
        alignmentAWrapped[alignmentAWrapped >= 0] = (
            alignmentAWrapped[alignmentAWrapped >= 0] / interpolation_factor
        ) % (period)
        alignmentB[alignmentB >= 0] = (
            alignmentB[alignmentB >= 0] / interpolation_factor
        ) % (template_period)
        roll_factor = (roll_factor / interpolation_factor) % (period)

        logger.info("roll_factor:\t{0}", roll_factor)
        logger.info("Aligned sequence #1 (wrapped):\t\t{0}", alignmentAWrapped)
        logger.info("Aligned sequence #2:\t\t\t{0}", alignmentB)

    # roll Alignment A, taking care of indels
    alignmentA = []
    indels = []
    for position in np.arange(alignmentAWrapped.shape[0]):
        if alignmentAWrapped[position] > -1:
            alignmentA.append((alignmentAWrapped[position] - roll_factor) % (period))
        else:
            idx = position - 1
            before = -1
            while before < 0 and idx < alignmentAWrapped.shape[0] - 1:
                before = alignmentAWrapped[(idx) % len(alignmentAWrapped)]
                idx = idx + 1
            indels.append(before)
    for indel in indels:
        alignmentA.insert(alignmentA.index(indel) + 1, -1)
    alignmentA = np.array(alignmentA)

    # get roll_factor properly
    roll_factor = get_roll_factor_at(alignmentA, alignmentB, ref_seq_phase)

    logger.info("roll_factor:\t{0}", roll_factor)
    logger.info("Aligned sequence #1 (unwrapped):\t", alignmentA)
    logger.info("Aligned sequence #2:\t\t\t", alignmentB)

    return alignmentA, alignmentB, roll_factor, score


# if __name__ == "__main__":
# roll = 7
# gap_penalty = 1
# shape = (1024, 1024)

# # Toy Example
# # toySequenceA and B have very slightly different rhythms but the same period
# toySequenceA = np.asarray(
#     [100, 150, 175, 200, 225, 230, 205, 180, 155, 120], dtype="uint8"
# )
# toySequenceA = np.roll(toySequenceA, -roll)
# periodA = toySequenceA.shape[0]
# toySequenceB = np.asarray(
#     [100, 125, 150, 175, 200, 225, 230, 205, 180, 120], dtype="uint8"
# )
# periodB = toySequenceB.shape[0]  # -0.5
# logger.info("Running toy example with:")
# logger.info("\tSequence A: ", toySequenceA)
# logger.info("\tSequence B: ", toySequenceB)

# # Make sequences 3D arrays (as expected for this algorithm)
# ndSequenceA = toySequenceA[:, np.newaxis, np.newaxis]
# ndSequenceB = toySequenceB[:, np.newaxis, np.newaxis]
# ndSequenceA = np.repeat(
#     np.repeat(ndSequenceA, shape[0], 1), shape[1], 2
# )  # make each frame actually 2D to check everything works for image frames
# ndSequenceB = np.repeat(np.repeat(ndSequenceB, shape[0], 1), shape[1], 2)

# alignmentA, alignmentB, roll_factor, score = cascading_needleman_wunsch(
#     ndSequenceA, ndSequenceB, periodA, periodB, gap_penalty=gap_penalty
# )
# logger.info("Roll factor: {0} (score: {1})", roll_factor, score)
# logger.info("Alignment Maps:")
# logger.info("\tMap A: {0}", alignmentA)
# logger.info("\tMap B: {0}", alignmentB)

# # Outputs for toy examples
# alignedSequenceA = []  # Create new lists to fill with aligned values
# alignedSequenceB = []
# for i in alignmentA:  # fill new sequence A
#     if i < 0:  # indel
#         alignedSequenceA.append(-1)
#     else:
#         alignedSequenceA.append(hlp.linear_interpolation(ndSequenceA, i)[0, 0])
# for i in alignmentB:  # fill new sequence B
#     if i < 0:  # indel
#         alignedSequenceB.append(-1)
#     else:
#         alignedSequenceB.append(hlp.linear_interpolation(ndSequenceB, i)[0, 0])

# # Print
# score = 0
# for i, j in zip(alignedSequenceA, alignedSequenceB):
#     if i > -1 and j > -1:
#         i = float(i)
#         j = float(j)
#         score = score - np.abs(i - j)
#     elif i > -1:
#         score = score - i
#     elif j > -1:
#         score = score - j
# logger.info("Aligned Sequences:")
# logger.info("\tMap A: {0}", alignmentA)
# logger.info("\tMap B: {0}", alignmentB)
# logger.info("Final Score: {0}", score)
