"""Modules for phase matching two sequences based on sequence alignment.
Uses a cascading form of the Needleman Wunsch algorithm.
This module includes all necessary functions."""

import numpy as np
import j_py_sad_correlation as jps
from loguru import logger

# Set-up logger
logger.disable("optical_gating_alignment")

# Import Numba or define njit if not available
try:
    from numba import njit

    logger.info("Numba loaded; just in time compilation will work.")
except Exception as e:

    def njit(func):
        return func

    logger.info("Numba was not loadable and so just in time compilation will not work.")


def get_roll_factor_at(alignment1, alignment2, target_phase1):
    # Log alignments for debugging
    logger.debug("alignment1: {0}", alignment1)
    logger.debug("alignment2: {0}", alignment2)
    logger.debug("target_phase1: {0}", target_phase1)

    # Convert lists to arrays (if needed)
    if isinstance(alignment1, list):
        alignment1 = np.array(alignment1)
    if isinstance(alignment2, list):
        alignment2 = np.array(alignment2)

    if int(target_phase1) > alignment1.max():  # is using int correct?
        # I don't believe this should happen in the real world
        logger.critical(
            "Target phase {0} greater than any phase in alignment1! {1}",
            target_phase1,
            alignment1,
        )

    # Search alignment1 for target_phase1
    if target_phase1 in alignment1:
        # Catch case where target_phase1 is exactly in alignment1
        logger.debug("Exact phase found in alignment1.")
        (target_index1,) = np.where(alignment1 == target_phase1)
        target_index1 = target_index1[0]  # just to get rid of the list
    else:
        # All other cases require interpolation
        logger.debug("No exact phase found in alignment1; Interpolating...")
        logger.trace("Lower: {0}", alignment1 < target_phase1)
        logger.trace("Upper: {0}", alignment1 < target_phase1)
        logger.trace("Gaps: {0}", alignment1 != -1)

        # Find all (non-gap) elements lower than the target phase
        (lower_index1,) = np.where(
            np.all([alignment1 < target_phase1, alignment1 != -1], axis=0)
        )
        logger.debug("All suitable lower_index1 candidates: {0}", lower_index1)

        # check there is only one contiguous stretch on indices
        # if there are more than one (expect 2)
        # then check which run is most likely to hold the correct lower_index1
        contiguous_indices = np.split(
            lower_index1, np.where(np.diff(lower_index1) > 1)[0] + 1
        )
        logger.trace(
            "Runs: {0}; Number of runs: {1};",
            contiguous_indices,
            len(contiguous_indices),
        )
        if len(contiguous_indices) > 1:
            for each_run in contiguous_indices:
                logger.trace("Phases for this run: {0}", alignment1[each_run])
                if alignment1[each_run[0]] < target_phase1 and (
                    alignment1[each_run[-1]] > target_phase1 - 1
                ):
                    lower_index1 = each_run
                    logger.debug("Taking a single contiguous run: {0}", lower_index1)
                    break  # we assume the earliest stretch is likely the best

        # take the last one from the run
        lower_index1 = lower_index1[-1]
        if lower_index1 >= len(alignment1) - 1:
            logger.debug("lower_index1 is at the end of alignment1")
            upper_index1 = lower_index1 + 1
            while alignment1[upper_index1 % len(alignment1)] == -1:
                upper_index1 = upper_index1 + 1
        else:
            # Find all (non-gap) elements greater than the target phase
            (upper_index1,) = np.where(
                np.all([alignment1 > target_phase1, alignment1 != -1], axis=0)
            )
            # catch issues caused by wrapping in alignmnent1
            if len(upper_index1[upper_index1 > lower_index1]) > 0:
                #  alignment1[0] != min and alignment1[-1] != min
                # Note: this method is safe to gaps
                upper_index1 = upper_index1[upper_index1 > lower_index1]
            logger.debug("All suitable upper_index1 candidates: {0}", upper_index1)
            # take the first one
            upper_index1 = upper_index1[0]
            # catch when lower_index1 is at the end and upper_index1 at the start
            # i.e. a wrap event
            if upper_index1 < lower_index1:
                upper_index1 = upper_index1 + len(alignment1)
                logger.debug("Unwrapping upper_index1.")

            if upper_index1 == len(alignment1):
                # I don't think this is a problem but I've kept it in for debugging
                logger.debug("upper_index1 is at the end of alignment1")

        lower_phase1 = alignment1[lower_index1 % len(alignment1)]
        upper_phase1 = alignment1[upper_index1 % len(alignment1)]
        # catch when lower_phase1 is at the end and upper_phase1 at the start
        # i.e. a wrap event
        if upper_phase1 < lower_phase1:
            upper_phase1 = upper_phase1 + alignment2.max() + 1
            logger.debug("Unwrapping upper_phase1.")

        # get interpolated index
        logger.debug(
            "Lower bound: {0} ({1}); Upper bound (unwrapped): {2} ({3})",
            lower_index1,
            lower_phase1,
            upper_index1,
            upper_phase1,
        )
        target_index1 = lower_index1 + (
            (target_phase1 - lower_phase1)
            * (upper_index1 - lower_index1)
            / (upper_phase1 - lower_phase1)
        ) % len(alignment1)

    logger.info(
        "Target phase of {0} at index {1} of alignment1", target_phase1, target_index1
    )

    target_index2 = None
    # Consider target_index in alignment2
    if np.isclose(target_index1 // 1, target_index1):
        # Catch case where target_index is, essentially, an integer
        logger.debug("Exact index used for alignment2.")
        target_phase2 = alignment2[int(target_index1) % len(alignment2)]
        if target_phase2 != -1:
            target_index2 = int(target_index1)
        else:
            logger.debug("Target index in alignment2 is a gap. Will use interpolation.")

    if target_index2 == None:
        # This situation requires interpolation or gap dealing
        logger.debug("No exact index found in alignment2; Interpolating...")
        target_index2 = target_index1.copy()

        # Get phase at index before
        lower_index2 = int(target_index1)
        lower_phase2 = alignment2[lower_index2 % len(alignment2)]
        while lower_phase2 == -1:
            # Deal with gaps
            logger.debug("lower_phase2 is a gap, decreasing lower_index2 by 1.")
            lower_index2 = lower_index2 - 1
            lower_phase2 = alignment2[lower_index2 % len(alignment2)]

        # Get phase at index after
        upper_index2 = int(target_index1 + 1)
        upper_phase2 = alignment2[upper_index2 % len(alignment2)]
        while upper_phase2 == -1:
            # Deal with gaps
            logger.debug("upper_phase2 is a gap, increasing upper_index2 by 1.")
            upper_index2 = upper_index2 + 1
            upper_phase2 = alignment2[upper_index2 % len(alignment2)]

        # catch when lower_phase2 is at the end and upper_phase2 at the start
        # i.e. a wrap event
        if upper_phase2 < lower_phase2:
            upper_phase2 = upper_phase2 + alignment2.max() + 1
            logger.debug("Unwrapping upper_phase2.")

        # Get interpolated phase
        logger.debug(
            "Lower bound: {0} ({1}); Upper bound (unwrapped): {2} ({3})",
            lower_index2,
            lower_phase2,
            upper_index2,
            upper_phase2,
        )
        target_phase2 = lower_phase2 + (
            (target_index2 - lower_index2)
            * (upper_phase2 - lower_phase2)
            / (upper_index2 - lower_index2)
        )

    logger.info(
        "New target phase of {0} at index {1} of alignment2",
        target_phase2,
        target_index2,
    )

    # calculate roll_factor
    roll_factor = (target_phase2 - target_phase1) % len(alignment1)
    logger.info("Roll_factor of {0}", roll_factor)

    return roll_factor


@njit
def fill_traceback_matrix(score_matrix, gap_penalty=0):
    """Using a score matrix, fill out a traceback matrix.
    This can then be traversed to identify a valid alignment.
    Note: this function contains a lot of basic python/numpy operations
    hence numba pre-compilation can speed things up
    but they have to occur in order, hence parallel=False
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
        (
            score_matrix.shape[0] + 1,
            score_matrix.shape[1] + 1,
            score_matrix.shape[1 - axis],
        ),
        dtype=np.float64,
    )

    # Create a new cascaded score array for each alignment (by rolling along axis)
    for n in np.arange(
        score_matrix.shape[1 - axis]
    ):  # the 1-axis tricks means we loop over 0 if axis=1 and vice versa
        logger.trace("Getting score matrix for roll of {0} frames...", n)
        cascades[:, :, n] = fill_traceback_matrix(score_matrix, gap_penalty=gap_penalty)
        score_matrix = np.roll(score_matrix, 1, axis=axis)

    return cascades


def traverse_traceback_matrix(sequence, template_sequence, traceback_matrix):
    """Traverse a tracbeack matrix and return aligned versions of two sequences.
    The template_sequence is less likely to have indels inserted."""
    if isinstance(sequence, list):
        sequence = np.array(sequence)
    if isinstance(template_sequence, list):
        template_sequence = np.array(template_sequence)

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
        logger.trace("-----")
        logger.trace(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "curr",
            x,
            y,
            traceback_matrix[x, y],
            sequence[-y, 0, 0],
            template_sequence[-x, 0, 0],
        )
        logger.trace(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "diag",
            x_up,
            y_left,
            traceback_matrix[x_up, y_left],
            sequence[-y_left, 0, 0],
            template_sequence[-x_up, 0, 0],
        )
        logger.trace(
            "{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});",
            "up  ",
            x_up,
            y,
            traceback_matrix[x_up, y],
            sequence[-y, 0, 0],
            template_sequence[-x_up, 0, 0],
        )
        logger.trace(
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
                logger.debug("Boundary Condition:\tI'm at the left")
                options[:] = [-np.inf, traceback_matrix[x_up, y], -np.inf]
        else:
            logger.debug("Boundary Condition:\tI'm at the top")
            if y_left >= 0:
                options[:] = [-np.inf, -np.inf, traceback_matrix[x, y_left]]
            else:
                logger.warning(
                    "Boundary Condition:\tI'm at the top left;\tI should not have got here!"
                )
                break
        direction = np.argmax(options)

        if direction == 1:
            alignmentA.append(-1)
            alignmentB.append(x_up)
            x = x_up
            logger.debug(
                "Direction Travelled:\tI've gone up; [{0}, {1}]",
                alignmentA[-1],
                alignmentB[-1],
            )
        elif direction == 0:
            alignmentA.append(y_left)
            alignmentB.append(x_up)
            x = x_up
            y = y_left
            logger.debug(
                "Direction Travelled:\tI've gone diagonal; [{0}, {1}]",
                alignmentA[-1],
                alignmentB[-1],
            )
        elif direction == 2:
            alignmentA.append(y_left)
            alignmentB.append(-1)
            y = y_left
            logger.debug(
                "Direction Travelled:\tI've gone left; [{0}, {1}]",
                alignmentA[-1],
                alignmentB[-1],
            )
        if x == 0 and y == 0:
            logger.info("Traversing Complete")
            traversing = False

    # TODO - is this right?
    # Treat leading and trailing tails as compresssible
    # I.e. **ABC and XYZ** == ABC and ZXY
    while alignmentA[0] == -1 and alignmentB[-1] == -1:
        del alignmentA[0], alignmentB[-1]
    while alignmentA[-1] == -1 and alignmentB[0] == -1:
        del alignmentA[-1], alignmentB[0]

    # Reverses sequence
    alignmentA = np.asarray(alignmentA[::-1], dtype=np.float)
    alignmentB = np.asarray(alignmentB[::-1], dtype=np.float)

    return alignmentA, alignmentB


def wrap_and_roll(alignmentA_wrapped, period, roll_factor):
    """Roll an alignment by roll_factor, taking care of indels."""
    # TODO - Do I actually care about the period?
    logger.trace("Wrapped alignment with gaps: {0}", alignmentA_wrapped)

    (gap_idx,) = np.where(alignmentA_wrapped == -1)

    alignmentA_unwrapped = np.roll(
        alignmentA_wrapped[alignmentA_wrapped > -1], -roll_factor
    )
    logger.trace("Unwrapped alignment without gaps: {0}", alignmentA_unwrapped)

    for gap in gap_idx:
        alignmentA_unwrapped = np.insert(alignmentA_unwrapped, gap, -1)
    logger.trace("Unwrapped alignment with gaps: {0}", alignmentA_unwrapped)

    return alignmentA_unwrapped


def cascading_needleman_wunsch(
    sequence,
    template_sequence,
    period=None,
    template_period=None,
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
    * ref_seq_phase: integer frame (in A) for which to return the roll factor
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

    # if interpolation_factor is not None:
    #     logger.info(
    #         "Interpolating by a factor of {0} for greater precision...",
    #         interpolation_factor,
    #     )
    #     logger.debug(sequence[:, 0, 0])
    #     sequence = hlp.interpolate_image_sequence(
    #         sequence, period, interpolation_factor=interpolation_factor
    #     )
    #     logger.debug(sequence[:, 0, 0])
    #     period = interpolation_factor * period
    #     logger.debug(template_sequence[:, 0, 0])
    #     template_sequence = hlp.interpolate_image_sequence(
    #         template_sequence,
    #         template_period,
    #         interpolation_factor=interpolation_factor,
    #     )
    #     logger.debug(template_sequence[:, 0, 0])
    #     template_period = interpolation_factor * template_period
    #     logger.info(
    #         "Sequence #1 now has period of {0} [{1}] frames and sequence #2 now has period of {2} [{3}] frames:",
    #         period,
    #         len(sequence),
    #         template_period,
    #         len(template_sequence),
    #     )

    # Calculate Score Matrix - C++
    # TODO remove these casts when jps can deal with lists
    score_matrix = jps.sad_grid(np.array(sequence), np.array(template_sequence))

    logger.debug("Score Matrix:")
    logger.debug(score_matrix)
    logger.debug(
        "Dtype: {0};\tShape: ({1},{2})",
        score_matrix.dtype,
        score_matrix.shape[0],
        score_matrix.shape[1],
    )

    # Cascade the SAD Grid
    cascades = construct_cascade(score_matrix, gap_penalty=gap_penalty, axis=0)
    logger.trace("Unrolled Traceback Matrix: {0}", cascades[:, :, 0])
    logger.trace(
        "Dtype: {0};\tShape: ({1},{2},{3})",
        cascades.dtype,
        cascades.shape[0],
        cascades.shape[1],
        cascades.shape[2],
    )

    # Pick Cascade and Roll sequence
    logger.debug("Cascades scores {0}", cascades[-1, -1, :])
    roll_factor = np.argmax(cascades[-1, -1, ::])
    logger.info(
        "Chose cascade {0} of {1} (global roll_factor of {2})",
        roll_factor + 1,
        cascades.shape[2],
        roll_factor,
    )

    traceback_matrix = cascades[:, :, roll_factor]
    logger.debug("Cascaded traceback matrix: {0}", traceback_matrix)

    score = traceback_matrix[-1, -1]
    # this score will be a max of zero (perfect alignment possible)
    # and a min of -np.iinfo(sequence.dtype).max, e.g. -(2^8 -1), * size
    score = -score / (sequence[0].size * cascades.shape[2])
    # this new score should be between 0.0 (good) and np.iinfo(sequence.dtype).max (bad)
    if score < 0:
        logger.critical("Negative Score")
        score = 0  # set to be terrible

    # logger.debug(sequence[:, 0, 0])
    sequence = np.roll(sequence, roll_factor, axis=0)
    # logger.debug(sequence[:, 0, 0])

    (alignmentAWrapped, alignmentB) = traverse_traceback_matrix(
        sequence, template_sequence, traceback_matrix
    )

    # TODO This is the wrong roll_factor (i.e-roll_factor)
    logger.debug("Rolled by (interpolated, global):\t{0}", roll_factor)
    logger.debug("Aligned sequence #1 (interpolated, wrapped):\t{0}", alignmentAWrapped)
    logger.trace("Aligned sequence #2 (interpolated):\t\t\t{0}", alignmentB)

    # roll Alignment A, taking care of indels
    alignmentA = wrap_and_roll(alignmentAWrapped, period, roll_factor)

    # get roll_factor properly
    roll_factor = get_roll_factor_at(alignmentA, alignmentB, ref_seq_phase)

    logger.debug("Alignment 1 (interpolated):\t{0}", alignmentA)
    logger.debug("Alignment 2 (interpolated):\t{0}", alignmentB)
    logger.info("Rolled by (interpolated, specific):\t{0}", roll_factor)
    logger.debug("Score: {0}", score)

    # # Undo interpolation
    # if interpolation_factor is not None:
    #     logger.info("De-interpolating for result...")
    #     # Divide by interpolation factor and modulo period
    #     # ignore -1s
    #     alignmentA[alignmentA >= 0] = (
    #         alignmentA[alignmentA >= 0] / interpolation_factor
    #     ) % (period)
    #     alignmentB[alignmentB >= 0] = (
    #         alignmentB[alignmentB >= 0] / interpolation_factor
    #     ) % (template_period)
    #     roll_factor = (roll_factor / interpolation_factor) % (period)

    #     logger.info("roll_factor:\t{0}", roll_factor)
    #     logger.info("Aligned sequence #1 (wrapped):\t\t{0}", alignmentA)
    #     logger.info("Aligned sequence #2:\t\t\t{0}", alignmentB)

    return roll_factor, score  # , alignmentA, alignmentB
