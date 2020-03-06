"""Sequence alignment algorithms for adaptive prospective optical gating.
Based on either cross correlation or cascading Needleman-Wunsch sequence
alignment of reference sets. Each new reference frame sequence should be
processed after determination. This module can correct for known drift
between sequences."""

import numpy as np
from loguru import logger
import helper as hlp
import cross_correlation as cc
import cascading_needleman_wunsch as cnw
import multipass_regression as mr

# Set-up logger
logger.disable("optical-gating-alignment")


def process_sequence(
    this_sequence,
    this_period,
    this_drift,
    sequence_history=None,
    period_history=None,
    drift_history=None,
    shift_history=None,
    algorithm="cnw",
    ref_seq_id=0,
    ref_seq_phase=0,
    max_offset=3,
    resampled_period=80,
    interp_factor=None,
    gap_penalty=0,
):
    """ Process a new reference sequence.
    Requires the historical information to be passed by the user.
    Returns new historical records that can be used in the next update.

    Inputs:
    * this_sequence: a PxMxN numpy array representing the new reference frames
      (or a list of numpy arrays representing the new reference frames)
    * this_period: the period for this_sequence (caller must determine this)
    * this_drift: the drift for this_sequence (caller must determine this)
      * if None no drift correction is used
    * sequence_history: a list of previous reference frame sets
    * period_history: a list of the previous periods for sequence_history
    * drift_history: a list of the previous drifts for sequence_history
      * if no drift correction is used, this is a dummy variable
    * shift_history: previously calculated relative shifts between
                     sequence_history
    * ref_seq_id: the index of sequence_history for which
                             ref_seq_phase applies
    * ref_seq_phase: the phase (index) we are trying to match in
                                ref_seq_id
    * resampled_period: the number of frames to use for resampled sequences
    * max_offset: how far apart historically to make comparisons
      * should be used to prevent comparing sequences that are far apart
        and have little similarity

    Outputs:
    * sequence_history: updated list of resampled reference frames
    * period_history: updated list of the periods for sequence_history
    * drift_history: updated list of the drifts for sequence_history
      * if no drift correction is used, this is a dummy variable
    * shift_history: updated list of shifts calculated for sequence_history
    * global_solution[-1]: roll factor for latest reference frames
    * residuals: residuals on least squares solution"""

    # Deal with this_sequence type
    if type(this_sequence) is list:
        this_sequence = np.vstack(this_sequence)

    # Check that the reference frames have a consistent shape
    for f in range(1, len(this_sequence)):
        if this_sequence[0].shape != this_sequence[f].shape:
            # There is a shape mismatch.
            logger.critical(
                "There is a shape mismatch within the new reference frames. \
                Frame 0: {0}; Frame {1}: {2}",
                this_sequence[0].shape,
                f,
                this_sequence[f].shape,
            )
            # Return an error message and code to indicate the problem.
            return (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                -1000.0,
                None,
                None,
                None,
            )
    # And that shape is compatible with the history that we already have
    if len(sequence_history) > 1:
        if this_sequence[0].shape != sequence_history[0][0].shape:
            # There is a shape mismatch.
            logger.critical(
                "There is a shape mismatch with historical reference \
                frames. Old shape: {1}; New shape: {2}",
                sequence_history[0][0].shape,
                this_sequence[0].shape,
            )
            # Return an error message and code to indicate the problem.
            return (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                -1000.0,
                None,
                None,
                None,
            )

    # Add latest reference frames to our sequence set
    this_resampled_sequence = hlp.interpolate_image_sequence(
        this_sequence, this_period, resampled_period / this_period
    )
    this_resampled_sequence = this_resampled_sequence.astype("uint8")  # FIXME: needed?
    sequence_history.append(this_resampled_sequence)
    period_history.append(resampled_period)

    if this_drift is not None:
        if len(drift_history) > 0:
            # Accumulate the drift throughout history
            drift_history.append(
                [
                    drift_history[-1][0] + this_drift[0],
                    drift_history[-1][1] + this_drift[1],
                ]
            )
        else:
            drift_history.append(this_drift)
    else:
        logger.warning(
            "No drift correction is being applied. This will seriously \
            impact phase locking."
        )

    # Update our shifts array.
    # Compare the current sequence with recent previous ones
    if len(sequence_history) > 1:
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(sequence_history) - max_offset - 1)
        for i in range(firstOne, len(sequence_history) - 1):
            logger.debug("--- {0} {1} ---", i, len(sequence_history) - 1)
            if i == ref_seq_id:
                logger.info("Using ref_seq_phase of {0}", ref_seq_phase)
                target = ref_seq_phase
            else:
                if this_drift is None:
                    if algorithm == "cc":
                        (alignment, target, score) = cc.rolling_cross_correlation(
                            sequence_history[ref_seq_id],
                            sequence_history[i],
                            resampled_period,
                            resampled_period,
                            target=ref_seq_phase,
                        )
                    elif algorithm == "cnw":
                        (alignment, target, score) = cnw.cascading_needleman_wunsch(
                            sequence_history[ref_seq_id],
                            sequence_history[i],
                            period_history[ref_seq_id],
                            period_history[i],
                            gap_penalty=gap_penalty,
                            interp_factor=interp_factor,
                            knownTargetFrame=ref_seq_phase,
                        )
                else:
                    drift = [
                        drift_history[i][0] - drift_history[ref_seq_id][0],
                        drift_history[i][1] - drift_history[ref_seq_id][1],
                    ]
                    seq1, seq2 = hlp.drift_correction(
                        sequence_history[ref_seq_id], sequence_history[i], drift
                    )
                    if algorithm == "cc":
                        (alignment, target, score) = cc.rolling_cross_correlation(
                            seq1,
                            seq2,
                            resampled_period,
                            resampled_period,
                            target=ref_seq_phase,
                        )
                    if algorithm == "cnw":
                        (alignment, target, score) = cnw.cascading_needleman_wunsch(
                            seq1,
                            seq2,
                            period_history[ref_seq_id],
                            period_history[i],
                            gap_penalty=gap_penalty,
                            interp_factor=interp_factor,
                            ref_seq_phase=ref_seq_phase,
                        )
                logger.info("Using target of {0}", target)
            if this_drift is None:
                if algorithm == "cc":
                    (alignment, roll_factor, score) = cc.rolling_cross_correlation(
                        sequence_history[i][:resampled_period],
                        this_resampled_sequence[:resampled_period],
                        resampled_period,
                        resampled_period,
                        target=target,
                    )
                elif algorithm == "cnw":
                    (alignment, roll_factor, score) = cnw.cascading_needleman_wunsch(
                        sequence_history[i],
                        sequence_history[-1],
                        period_history[i],
                        period_history[-1],
                        gap_penalty=gap_penalty,
                        interp_factor=interp_factor,
                        knownTargetFrame=target,
                    )
            else:
                seq1, seq2 = hlp.drift_correction(
                    sequence_history[i], sequence_history[-1], this_drift
                )
                if algorithm == "cc":
                    (alignment1, alignment2, roll_factor, score) = cc.rolling_cross_correlation(
                        seq1, seq2, resampled_period, resampled_period, target=target
                    )
                elif algorithm == "cnw":
                    (alignment1, alignment2 roll_factor, score) = cnw.cascading_needleman_wunsch(
                        seq1,
                        seq2,
                        period_history[i],
                        period_history[-1],
                        gap_penalty=gap_penalty,
                        interp_factor=interp_factor,
                        knownTargetFrame=target,
                    )

            # Append to history
            if algorithm == "cc":
                shift_history.append(
                    (
                        i,
                        len(sequence_history) - 1,
                        (roll_factor - target) % resampled_period,
                        score,
                    )
                )
            elif algorithm == "cnw":
                shift_history.append(
                    (i, len(sequence_history) - 1, roll_factor - target, 1)
                )  # add score here

    logger.debug("Printing shifts:")
    logger.debug(shift_history)

    global_solution = mr.MakeShiftsSelfConsistent(
        shift_history,
        len(sequence_history),
        period_history,
        ref_seq_id=ref_seq_id,
        ref_seq_phase=ref_seq_phase,
    )

    logger.debug("Solution:")
    logger.debug(global_solution)

    logger.info("Reference Frame rolling by: {0}", global_solution[-1])

    # Catch for outputs on first period
    if len(sequence_history) == 1:
        score = 0
        alignment1 = []

    # Count indels in last returned alignment
    indels = np.sum(alignment1 == -1)

    # Note for developers:
    # there are two other return statements in this function
    return (
        sequence_history,
        period_history,
        drift_history,
        shift_history,
        global_solution[-1],
        score,
        indels,
    )


# if __name__ == "__main__":
#     logger.enable("optical-gating-alignment")
#     logger.warning("This test is still broken.")
#     numStacks = 10
#     stackLength = 10
#     width = 10
#     height = 10

#     sequence_historyDrift = []
#     periodHistoryDrift = []
#     shiftsDrift = []
#     driftHistoryDrift = []

#     sequence_history = []
#     periodHistory = []
#     shifts = []
#     driftHistory = []

#     for i in range(numStacks):
#         logger.info("Running for stack {0}", i)
#         # Make new toy sequence
#         thisPeriod = stackLength - 0.5
#         seq1 = (
#             np.arange(stackLength) + np.random.randint(0, stackLength + 1)
#         ) % thisPeriod
#         logger.info("New Sequence: {0}; Period: {1} ({2})", seq1, thisPeriod, len(seq1))
#         seq2 = np.asarray(seq1, "uint8").reshape([len(seq1), 1, 1])
#         seq2 = np.repeat(np.repeat(seq2, width, 1), height, 2)

#         # Run MCC without Drift
#         (
#             sequence_history,
#             periodHistory,
#             driftHistory,
#             shifts,
#             roll_factor,
#             score,
#             indels,
#         ) = process_sequence(
#             seq2,
#             thisPeriod,
#             None,
#             sequence_history,
#             periodHistory,
#             driftHistory,
#             shifts,
#             gap_penalty=0,
#             ref_seq_phase=0,
#             resampled_period=80,
#             max_offset=3,
#         )

#         # Outputs for toy examples
#         seqOut = (seq1 + roll_factor) % thisPeriod
#         logger.info("Aligned Sequence (wout Drift): {0}", seqOut)

#         # Run MCC with Drift of [0,0]
#         (
#             sequence_historyDrift,
#             periodHistoryDrift,
#             driftHistoryDrift,
#             shiftsDrift,
#             roll_factor,
#         ) = process_sequence(
#             seq2,
#             thisPeriod,
#             [0, 0],
#             sequence_historyDrift,
#             periodHistoryDrift,
#             driftHistoryDrift,
#             shiftsDrift,
#             gap_penalty=0,
#             ref_seq_phase=0,
#             resampled_period=80,
#             max_offset=3,
#         )

#         # Outputs for toy examples
#         seqOut = (seq1 + roll_factor) % thisPeriod
#         logger.info("Aligned Sequence (with Drift): {0}", seqOut)
