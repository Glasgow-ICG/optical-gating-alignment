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
    logger.debug("{0} {1} {2}", alignment1, alignment2, phase1)

    if isinstance(alignment1, list):
        alignment1 = np.array(alignment1)
    if isinstance(alignment2, list):
        alignment2 = np.array(alignment2)

    length1 = len(alignment1)
    length2 = len(alignment2)

    # First get the exact index of phase1 in alignment1
    # Case 1: where phase1 is in alignment1
    # i.e. phase1 likely to be a whole number
    if phase1 in alignment1:
        # DEVNOTE: we assume there is only ever one, which should be true or something has gone awry
        idxPos = np.nonzero(alignment1 == phase1)[0][0]
        logger.info("Exact phase {0} found at index {1} in alignment 1", phase1, idxPos)
    # Case 2: where phase1 is not in alignment1
    # find the indices in alignment1 that are the lower and upper bounds
    # then use linear interpolation to get the exact index
    else:
        logger.info(
            "Exact phase {0} not found in alignment 1; searching for bounds.", phase1
        )
        # set starting bounds to a non-index
        lower_bound = -1
        upper_bound = -1
        # DEVNOTE: the allow flag deals with the scenario that my alignment starts after the desired phase, i.e. alignment[0]>phase1, and wraps latter in the sequence
        allow = False  # only turn on if I've seen a value smaller than desired
        for idx1 in range(length1):
            logger.debug(
                "allow := {0};\tidx := {1};\talignment1[idx1] := {2};\t(alignment1[idx1]>=0) := {3};\t >= min(alignment1) := {4};\t > phase1 := {5}",
                allow,
                idx1,
                alignment1[idx1],
                alignment1[idx1] >= 0,
                alignment1[idx1] == min(alignment1[alignment1 >= 0]),
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
                and alignment1[idx1] == min(alignment1[alignment1 >= 0])
                and alignment1[idx1] > phase1
            ):
                # not gap and not started and is smallest in alignment1
                # but is greater than phase, i.e. target is between the wrap point
                logger.info("Desired phase at alignment sequence 1 wrap point (type1)")
                # set lower and upper bound and stop searching
                lower_bound = idx1 - 1
                upper_bound = idx1
                logger.debug(
                    "Preliminary bounds applied: {0} [{1}], {2} [{3}] for {4} (type1)",
                    lower_bound,
                    alignment1[lower_bound],
                    upper_bound,
                    alignment1[upper_bound],
                    phase1,
                )
                break
            elif allow and alignment1[idx1] > phase1:
                # started and higher than phase1 (not gap implied)
                lower_bound = idx1 - 1
                upper_bound = idx1
                logger.debug(
                    "Preliminary bounds applied: {0} [{1}], {2} [{3}] for {4} (type1)",
                    lower_bound,
                    alignment1[lower_bound],
                    upper_bound,
                    alignment1[upper_bound],
                    phase1,
                )
                break
            elif allow and alignment1[idx1] == 0:
                # started and wrap point (not gap implied)
                # type2 wrap - set allow earlier but have now reached 0 value
                # in alignment1 string and not found an alignment greater than
                # phase1, i.e. presume it hovers between the highest value and the 0.
                logger.info("Desired phase at alignment sequence 1 wrap point (type2)")
                lower_bound = idx1 - 1
                upper_bound = idx1
                logger.debug(
                    "Preliminary bounds applied: {0} [{1}], {2} [{3}] for {4} (type1)",
                    lower_bound,
                    alignment1[lower_bound],
                    upper_bound,
                    alignment1[upper_bound],
                    phase1,
                )
                break

        if lower_bound < 0 and upper_bound < 0 and not allow:
            # no valid bounds set
            # assume phase1 completely not in alignment then
            # this should never happen so I don't have a test for it
            logger.critical("Phase not found in alignment sequence 1")
            logger.debug(phase1, alignment1)
            return None

        if lower_bound < 0 and upper_bound < 0 and allow:
            # started searching but didn't set bounds
            # assume bounds are around the wrap point
            # this type of wrap point is when alignment 1
            # ends lower than it starts, i.e. the wrap point
            # of the ids sequence is in the middle of the actual string
            logger.info("Wrapping around alignment sequence 1")
            lower_bound = idx1
            upper_bound = idx1 + 1
            logger.debug(
                "Preliminary bounds applied: {0} [{1}], {2} [{3}] for {4} (type1)",
                lower_bound,
                alignment1[lower_bound],
                upper_bound,
                alignment1[upper_bound % length1],
                phase1,
            )

        # account for gaps in upper bound
        while alignment1[upper_bound % length1] < 0:
            logger.debug("Increasing upper bound by one due to indel.")
            upper_bound = upper_bound + 1
        # account for gaps in lower bound
        while alignment1[lower_bound % length1] < 0:
            logger.debug("Decreasing lower bound by one due to indel.")
            lower_bound = lower_bound - 1

        logger.info(
            "Interpolating with lower bound of {0} and upper bound of {1}",
            lower_bound,
            upper_bound % length1,
        )

        logger.debug(
            "{0} {1} {2} {3} {4}",
            lower_bound,
            upper_bound,
            length1,
            lower_bound % length1,
            upper_bound % length1,
        )
        if (
            upper_bound % length1 == upper_bound
            and lower_bound % length1 == lower_bound
            and alignment1[upper_bound % length1] > alignment1[lower_bound % length1]
        ):
            logger.debug(
                "Both upper and lower bounds within period1 and no wrapping in idx occurred."
            )
            interpolated_index1 = (phase1 - alignment1[lower_bound]) / (
                alignment1[upper_bound] - alignment1[lower_bound]
            )
            logger.debug(
                "interpolated index {0} = ({1} - {2}) / ({3} - {4}) = {5} / {6};",
                interpolated_index1,
                phase1,
                alignment1[lower_bound],
                alignment1[upper_bound],
                alignment1[lower_bound],
                (phase1 - alignment1[lower_bound]),
                (alignment1[upper_bound] - alignment1[lower_bound]),
            )
        elif (
            upper_bound % length1 == upper_bound
            and lower_bound % length1 == lower_bound
            and not alignment1[upper_bound % length1]
            > alignment1[lower_bound % length1]
        ):
            logger.debug(
                "Both upper and lower bounds within period1 but wrapping in idx occurred at upper bound."
            )
            interpolated_index1 = (phase1 - alignment1[lower_bound]) / (
                alignment1.max() + 1 + alignment1[upper_bound] - alignment1[lower_bound]
            )
            logger.debug(
                "interpolated index {0} = ({1} - {2}) / ({3} + {4} -{5}) = {6} / {7};",
                interpolated_index1,
                phase1,
                alignment1[lower_bound],
                alignment1.max() + 1,
                alignment1[upper_bound],
                alignment1[lower_bound],
                (phase1 - alignment1[lower_bound]),
                (alignment1.max() + alignment1[upper_bound] - alignment1[lower_bound]),
            )
        elif (
            upper_bound % length1 == upper_bound
            and not lower_bound % length1 == lower_bound
            and alignment1[upper_bound % length1] > alignment1[lower_bound % length1]
        ):
            logger.debug(
                "Only upper bound within period1 and no wrapping in idx occurred."
            )
            interpolated_index1 = (
                phase1 - alignment1[lower_bound % length1] + length1
            ) / (alignment1[upper_bound] - alignment1[lower_bound % length1])
            logger.debug(
                "interpolated index {0} = ({1} - {2}) / ({3} - {4}) = {5} / {6};",
                interpolated_index1,
                phase1,
                alignment1[lower_bound % length1],
                alignment1[upper_bound],
                alignment1[lower_bound % length1],
                (phase1 - alignment1[lower_bound % length1]),
                (alignment1[upper_bound] - alignment1[lower_bound % length1]),
            )
        elif (
            upper_bound % length1 == upper_bound
            and not lower_bound % length1 == lower_bound
            and not alignment1[upper_bound % length1]
            > alignment1[lower_bound % length1]
        ):
            logger.debug(
                "Only upper bound within period but wrapping in idx occurred at upper bound."
            )
            interpolated_index1 = (
                phase1 - alignment1[lower_bound % length1] + length1
            ) / (
                alignment1.max()
                + 1
                + alignment1[upper_bound]
                - alignment1[lower_bound % length1]
            )
            logger.debug(
                "interpolated index {0} = ({1} - {2}) / ({3} + {4} - {5}) = {6} / {7};",
                interpolated_index1,
                phase1,
                alignment1[lower_bound % length1],
                alignment1.max(),
                alignment1[upper_bound],
                alignment1[lower_bound % length1],
                (phase1 - alignment1[lower_bound % length1]),
                (
                    alignment1.max()
                    + 1
                    + alignment1[upper_bound]
                    - alignment1[lower_bound % length1]
                ),
            )
        elif (
            not upper_bound % length1 == upper_bound
            and lower_bound % length1 == lower_bound
            and alignment1[upper_bound % length1] > alignment1[lower_bound % length1]
        ):
            logger.debug(
                "Only lower bound within period1 and no wrapping in idx occurred."
            )
            interpolated_index1 = (phase1 - alignment1[lower_bound]) / (
                alignment1[upper_bound % length1] - alignment1[lower_bound]
            )
            logger.debug(
                "interpolated index {0} = ({1} - {2}) / ({4} - {5}) = {6} / {7};",
                interpolated_index1,
                phase1,
                alignment1[lower_bound],
                length1,
                alignment1[upper_bound % length1],
                alignment1[lower_bound],
                (phase1 - alignment1[lower_bound]),
                (alignment1[upper_bound % length1] - alignment1[lower_bound]),
            )
        elif (
            not upper_bound % length1 == upper_bound
            and lower_bound % length1 == lower_bound
            and not alignment1[upper_bound % length1]
            > alignment1[lower_bound % length1]
        ):
            logger.debug(
                "Only lower bound within period1 but wrapping in idx occurred at upper bound."
            )
            interpolated_index1 = (phase1 - alignment1[lower_bound]) / (
                +length1 + alignment1[upper_bound % length1] - alignment1[lower_bound]
            )
            logger.debug(
                "interpolated index {0} = ({1} - {2}) / ({3} + {4} - {5}) = {6} / {7};",
                interpolated_index1,
                phase1,
                alignment1[lower_bound],
                length1,
                alignment1[upper_bound % length1],
                alignment1[lower_bound],
                (phase1 - alignment1[lower_bound]),
                (alignment1[upper_bound % length1] - alignment1[lower_bound]),
            )
        else:
            pass
            logger.critical("I don't think this should ever be reached!")

        idxPos = (interpolated_index1 * (upper_bound - lower_bound)) + lower_bound

        logger.info("Phase positioned at interpolated index {0} in alignment 1", idxPos)

        logger.debug(
            "phase1 := {0};\tlower_bound := {1};\talignment1 := {2};\tupper_bound := {3};\t length1 := {4};\t(upper_bound %% length1) := {5};\talignment1 := {6};\tinterpolated_index1 :={7};",
            phase1,
            lower_bound,
            alignment1[lower_bound],
            upper_bound,
            length1,
            upper_bound % length1,
            alignment1[upper_bound % length1],
            interpolated_index1,
        )

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
        logger.info(
            "Exact index {0} used in alignment 2 [{1}] to match a phase of {2}",
            idxPos,
            phase2,
            phase1,
        )

    else:
        alignment2_lower_bound = int(idxPos)  # same as np.floor
        alignment2_upper_bound = int(idxPos + 1)  # same as np.ceil
        logger.debug("{0} {1}", alignment2_lower_bound, alignment2_upper_bound)
        logger.debug("{0} {1}", length1, length2)

        # check not same value (occurs when exactly hits an index)
        if alignment2_lower_bound == alignment2_upper_bound:
            logger.debug("Lower and upper bounds equal; increasing upper bound.")
            alignment2_upper_bound = alignment2_upper_bound + 1

        # deal with gaps in alignment2
        while alignment2[int(alignment2_lower_bound % length2)] < 0:
            alignment2_lower_bound = alignment2_lower_bound - 1
        while alignment2[int(alignment2_upper_bound % length2)] < 0:
            alignment2_upper_bound = alignment2_upper_bound + 1

        # interpolate for exact position
        logger.debug(
            "Interpolating with lower bound of {0} and upper bound of {1}",
            alignment2_lower_bound % length2,
            alignment2_upper_bound % length2,
        )

        # FIXME
        if (
            alignment2_upper_bound % length2 == alignment2_upper_bound
            and alignment2_lower_bound % length2 == alignment2_lower_bound
            and alignment2[alignment2_upper_bound % length2]
            > alignment2[alignment2_lower_bound % length2]
        ):
            logger.debug(
                "Both upper and lower bounds within period and no wrapping occurred."
            )
            interpolated_index2 = (idxPos - alignment2_lower_bound) / (
                alignment2_upper_bound - alignment2_lower_bound
            )
            logger.debug(
                "interpolated index2 {0} = ({1} - {2}) / ({3} - {4}) = {5} / {6};",
                interpolated_index2,
                idxPos,
                alignment2_lower_bound,
                alignment2_upper_bound,
                alignment2_lower_bound,
                (idxPos - alignment2_lower_bound),
                (alignment2_upper_bound - alignment2_lower_bound),
            )
            phase2 = (
                interpolated_index2 * (alignment2[alignment2_upper_bound] + 1)
            ) + alignment2[alignment2_lower_bound]
            logger.debug(
                "phase2 {0} = {1} * ({2} + {3} + 1) + {4} = {5} * {6} + {7};",
                phase2,
                interpolated_index2,
                alignment2.max(),
                alignment2[alignment2_upper_bound % length2],
                alignment2[alignment2_lower_bound],
                interpolated_index2,
                (alignment2.max() + alignment2[alignment2_upper_bound] + 1),
                alignment2[alignment2_lower_bound],
            )
        elif (
            alignment2_upper_bound % length2 == alignment2_upper_bound
            and alignment2_lower_bound % length2 == alignment2_lower_bound
            and not alignment2[alignment2_upper_bound % length2]
            > alignment2[alignment2_lower_bound % length2]
        ):
            logger.debug(
                "Both upper and lower bounds within period but wrapping occurred in alignment2_upper_bound."
            )
            interpolated_index2 = (idxPos - alignment2_lower_bound) / (
                alignment2_upper_bound - alignment2_lower_bound
            )
            logger.debug(
                "interpolated index2 {0} = ({1} - {2}) / ({3} - {4}) = {5} / {6};",
                interpolated_index2,
                idxPos,
                alignment2_lower_bound,
                alignment2_upper_bound,
                alignment2_lower_bound,
                (idxPos - alignment2_lower_bound),
                (alignment2_upper_bound - alignment2_lower_bound),
            )
            phase2 = (
                interpolated_index2
                * (alignment2.max() + alignment2[alignment2_upper_bound] + 1)
            ) + alignment2[alignment2_lower_bound]
            logger.debug(
                "phase2 {0} = {1} * ({2} + {3} + 1) + {4} = {5} * {6} + {7};",
                phase2,
                interpolated_index2,
                alignment2.max(),
                alignment2[alignment2_upper_bound % length2],
                alignment2[alignment2_lower_bound],
                interpolated_index2,
                (alignment2.max() + alignment2[alignment2_upper_bound] + 1),
                alignment2[alignment2_lower_bound],
            )
        elif (
            alignment2_upper_bound % length2 == alignment2_upper_bound
            and not alignment2_lower_bound % length2 == alignment2_lower_bound
            and alignment2[alignment2_upper_bound % length2]
            > alignment2[alignment2_lower_bound % length2]
        ):
            logger.debug("Only upper bound within period and no wrapping occurred.")
            interpolated_index2 = (idxPos - alignment2_lower_bound + length2) / (
                alignment2_upper_bound - alignment2_lower_bound
            )
            logger.debug(
                "interpolated index2 {0} = ({1} - {2} + {3}) / ({4} - {5}) = {6} / {7};",
                interpolated_index2,
                idxPos,
                alignment2_lower_bound,
                length2,
                alignment2_upper_bound,
                alignment2_lower_bound,
                (idxPos - alignment2_lower_bound),
                (alignment2_upper_bound - alignment2_lower_bound),
            )
            phase2 = (
                interpolated_index2 * (alignment2[alignment2_upper_bound] + 1)
            ) + alignment2[alignment2_lower_bound]
            logger.debug(
                "phase2 {0} = {1} * ({2} + {3} + 1) + {4} = {5} * {6} + {7};",
                phase2,
                interpolated_index2,
                alignment2.max(),
                alignment2[alignment2_upper_bound % length2],
                alignment2[alignment2_lower_bound],
                interpolated_index2,
                (alignment2.max() + alignment2[alignment2_upper_bound] + 1),
                alignment2[alignment2_lower_bound],
            )
        elif (
            alignment2_upper_bound % length2 == alignment2_upper_bound
            and not alignment2_lower_bound % length2 == alignment2_lower_bound
            and not alignment2[alignment2_upper_bound % length2]
            > alignment2[alignment2_lower_bound % length2]
        ):
            logger.debug(
                "Only upper bound within period but wrapping occurred in upper bound."
            )
            interpolated_index2 = (idxPos - alignment2_lower_bound + length2) / (
                alignment2_upper_bound - alignment2_lower_bound
            )
            logger.debug(
                "interpolated index2 {0} = ({1} - {2} + {3}) / ({4} - {5}) = {6} / {7};",
                interpolated_index2,
                idxPos,
                alignment2_lower_bound,
                length2,
                alignment2_upper_bound,
                alignment2_lower_bound,
                (idxPos - alignment2_lower_bound),
                (alignment2_upper_bound - alignment2_lower_bound),
            )
            phase2 = (
                interpolated_index2
                * (alignment2.max() + alignment2[alignment2_upper_bound] + 1)
            ) + alignment2[alignment2_lower_bound]
            logger.debug(
                "phase2 {0} = {1} * ({2} + {3} + 1) + {4} = {5} * {6} + {7};",
                phase2,
                interpolated_index2,
                alignment2.max(),
                alignment2[alignment2_upper_bound % length2],
                alignment2[alignment2_lower_bound],
                interpolated_index2,
                (alignment2.max() + alignment2[alignment2_upper_bound] + 1),
                alignment2[alignment2_lower_bound],
            )
        elif (
            not alignment2_upper_bound % length2 == alignment2_upper_bound
            and alignment2_lower_bound % length2 == alignment2_lower_bound
            and alignment2[alignment2_upper_bound % length2]
            > alignment2[alignment2_lower_bound % length2]
        ):
            logger.debug("Only lower bound within period and no wrapping.")
            interpolated_index2 = (idxPos - alignment2_lower_bound) / (
                alignment2_upper_bound - alignment2_lower_bound
            )
            logger.debug(
                "interpolated index2 {0} = ({1} - {2}) / ({4} - {5}) = {6} / {7};",
                interpolated_index2,
                idxPos,
                alignment2_lower_bound,
                length2,
                alignment2_upper_bound,
                alignment2_lower_bound,
                (idxPos - alignment2_lower_bound),
                (alignment2_upper_bound - alignment2_lower_bound),
            )
            phase2 = (
                interpolated_index2 * (alignment2[alignment2_upper_bound % length2] + 1)
            ) + alignment2[alignment2_lower_bound]
            logger.debug(
                "phase2 {0} = {1} * ({2} + {3} + 1) + {4} = {5} * {6} + {7};",
                phase2,
                interpolated_index2,
                alignment2.max(),
                alignment2[alignment2_upper_bound % length2],
                alignment2[alignment2_lower_bound],
                interpolated_index2,
                (alignment2.max() + alignment2[alignment2_upper_bound % length2] + 1),
                alignment2[alignment2_lower_bound],
            )
        elif (
            not alignment2_upper_bound % length2 == alignment2_upper_bound
            and alignment2_lower_bound % length2 == alignment2_lower_bound
            and not alignment2[alignment2_upper_bound % length2]
            > alignment2[alignment2_lower_bound % length2]
        ):
            logger.debug(
                "Only lower bound within period but wrapping occurred in upper bound."
            )
            interpolated_index2 = (idxPos - alignment2_lower_bound) / (
                alignment2_upper_bound - alignment2_lower_bound
            )
            logger.debug(
                "interpolated index2 {0} = ({1} - {2}) / ({4} - {5}) = {6} / {7};",
                interpolated_index2,
                idxPos,
                alignment2_lower_bound,
                length2,
                alignment2_upper_bound,
                alignment2_lower_bound,
                (idxPos - alignment2_lower_bound),
                (alignment2_upper_bound - alignment2_lower_bound),
            )
            phase2 = (
                interpolated_index2
                * (alignment2.max() + alignment2[alignment2_upper_bound % length2] + 1)
            ) + alignment2[alignment2_lower_bound]
            logger.debug(
                "phase2 {0} = {1} * ({2} + {3} + 1) + {4} = {5} * {6} + {7};",
                phase2,
                interpolated_index2,
                alignment2.max(),
                alignment2[alignment2_upper_bound % length2],
                alignment2[alignment2_lower_bound],
                interpolated_index2,
                (alignment2.max() + alignment2[alignment2_upper_bound % length2] + 1),
                alignment2[alignment2_lower_bound],
            )
        else:
            pass
            logger.critical("I don't think this should ever be reached!")

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

    if phase2 is None:
        logger.critical("No phase calculated for alignment sequence 2")

    roll_factor = (phase1 - phase2) % length2  # TODO - why?!

    return roll_factor


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
        score_matrix = roll_score_matrix(score_matrix, -1, axis=axis)

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


def wrap_and_roll(alignmentAWrapped, period, roll_factor):
    """Roll an alignment by roll_factor, taking care of indels."""
    alignmentA = []
    indels = []
    logger.debug("Wrapped alignment with indels: {0}", alignmentAWrapped)
    logger.debug("Period: {0}; Roll_factor: {1}", period, roll_factor)
    for position in np.arange(alignmentAWrapped.shape[0]):
        # for each alignment idx
        if alignmentAWrapped[position] > -1:
            # if not gap, add to output
            alignmentA.append((alignmentAWrapped[position] - roll_factor) % (period))
        else:
            # if gap, work out the value to put the indel before
            idx = position - 1
            before = -1
            while before < 0 and (idx % period) < alignmentAWrapped.shape[0]:
                before = alignmentAWrapped[(idx) % len(alignmentAWrapped)]
                idx = (idx + 1) % period
            indels.append(before)
    logger.debug("Unwrapped alignment with no indels: {0}", alignmentA)
    # go through the indels from end to start and insert them
    for indel in indels[::-1]:
        alignmentA.insert(alignmentA.index(indel) + 1, -1)
    alignmentA = np.array(alignmentA)
    logger.debug("Unwrapped alignment with indels: {0}", alignmentA)
    return alignmentA


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
    score_matrix = jps.sad_grid(sequence, template_sequence)

    logger.debug("Score Matrix:")
    logger.debug(score_matrix)
    logger.debug(
        "Dtype: {0};\tShape: ({1},{2})",
        score_matrix.dtype,
        score_matrix.shape[0],
        score_matrix.shape[1],
    )

    # Cascade the SAD Grid
    cascades = construct_cascade(score_matrix, gap_penalty=gap_penalty, axis=1)
    logger.debug("Unrolled Traceback Matrix:")
    logger.debug(cascades[:, :, 0])
    logger.debug(
        "Dtype: {0};\tShape: ({1},{2},{3})",
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
    if score < 0:
        logger.warning("Negative Score")
        score = 0

    traceback_matrix = cascades[:, :, roll_factor]
    sequence = np.roll(sequence, roll_factor, axis=0)
    logger.debug("Cascades scores {0}", cascades[-1, -1, :])
    logger.info("Chose cascade {0} of {1}:", roll_factor+1, len(sequence))
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

    logger.info("roll_factor (interpolated, base):\t{0}", roll_factor)
    logger.info("Aligned sequence #1 (interpolated, wrapped):\t{0}", alignmentAWrapped)
    logger.info("Aligned sequence #2 (interpolated):\t\t\t{0}", alignmentB)

    # roll Alignment A, taking care of indels
    alignmentA = wrap_and_roll(alignmentAWrapped, period, roll_factor)

    # get roll_factor properly
    roll_factor = get_roll_factor_at(alignmentA, alignmentB, ref_seq_phase)

    logger.info("roll_factor (interpolated, specific):\t{0}", roll_factor)
    logger.info("Aligned sequence #1 (interpolated, unwrapped):\t{0}", alignmentA)
    logger.info("Aligned sequence #2 (interpolated):\t\t\t{0}", alignmentB)

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

    return alignmentA, alignmentB, roll_factor, score
