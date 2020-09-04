"""Sequence alignment algorithms for adaptive prospective optical gating.
Based on either cross correlation or cascading Needleman-Wunsch sequence
alignment of reference sets. Each new reference frame sequence should be
processed after determination. This module can correct for known drift
between sequences."""

import sys
import numpy as np
from loguru import logger
from . import helper as hlp
from . import cross_correlation as cc
from . import cascading_needleman_wunsch as cnw
from . import multipass_regression as mr

# Set-up logger
logger.disable("optical_gating_alignment")  # turn off the module logger (default)


def set_logger(level="CRITICAL", format=None):
    """Small helper to change logger level."""
    logger.enable("optical_gating_alignment")
    logger.remove()
    if format == "jupyter":
        logger.add(sys.stdout, level=level, format="{message}")
    elif format is not None:
        logger.add(sys.stderr, level=level, format=format)
    else:
        logger.add(sys.stderr, level=level)


def process_sequence(
    this_sequence,
    this_period,
    this_drift,
    sequence_history=None,
    period_history=None,
    drift_history=None,
    shift_history=None,
    global_solution=None,
    max_offset=3,
    ref_seq_id=0,
    ref_seq_phase=0,
    algorithm="cc",
    **kwargs
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
    * max_offset: how far apart historically to make comparisons
    * for 'cc' algorithm:
    * method
    * resampled_period: the number of frames to use for resampled sequences
      * should be used to prevent comparing sequences that are far apart
        and have little similarity
    * for 'cnw' alorithm:
    * ref_seq_id: the index of sequence_history for which
                             ref_seq_phase applies
    * ref_seq_phase: the phase (index) we are trying to match in
                                ref_seq_id

    Outputs:
    * sequence_history: updated list of resampled reference frames
    * period_history: updated list of the periods for sequence_history
    * drift_history: updated list of the drifts for sequence_history
      * if no drift correction is used, this is a dummy variable
    * shift_history: updated list of shifts calculated for sequence_history
    * global_solution[-1]: roll factor for latest reference frames
    
    Philosophy for "cc" (cross correlation) algorithm:
    * Given a target (ref_seq_phase) in an existing period (ref_seq_id, usually the first in a sequence)
    * Convert target to a shared base (resampled_period, usually 80)
    * Take new period (this_period)
    * Resample this image sequence to have number of frames equal to the shared base (this_resampled_sequence)
    * Pairwise cross correlate between the new, resampled period and historical (also resampled) periods
    * Return the minimum cross-correlation for each pair (the roll_factor)
    * Use multi-pass regression to use historical pairwise comparisons to minimise errors in the latest roll factor and get a global_roll_factor
    * Use the global roll factor (modulo the shared base) and the initial target to identify a new target frame
    * Convert the new target frame into the original period

    Philosophy for "cnw" (cascading Needleman Wunsch) algorithm:
    * Given a target (ref_seq_phase) in an existing period (ref_seq_id, usually the first in a sequence)
      * DO NOT convert target to a shared base as this should not be necessary for this algorithm
      * TODO: Interpolate sequence for higher precision (optional)
    * Take new period (this_period)
      * DO NOT resample this image sequence
    * Cascading Needleman Wunsch alignment between the new, resampled period and historical periods
      * IMPORTANT: The target (ref_seq_phase) must be followed throughout these pairwise comparisons for accurate alignment
    * CURRENTLY: Return the best alignment for each pair (the roll_factor)
    * TODO: Return the best alignment for each pair (alignment1, alignment2); Return the specific roll_factor for the alignments at the target (e.g. ref_seq_phase at ref_seq_id)
    * Use multi-pass regression to use historical pairwise comparisons to minimise errors in the latest roll factor and get a global_roll_factor
    * Use the global roll factor (modulo the shared base) and the initial target to identify a new target frame
      * TODO: De-interpolate the target if required (optional)
    """

    # Deal with this_sequence type - mostly legacy
    #if type(this_sequence) is list:
        #this_sequence = np.vstack(this_sequence)

    logger.debug("Parsing algorithm-specific options ({0}):".format(algorithm))
    options = kwargs.keys()
    for option in options:
        if option not in [
            "resampled_period",
            "method",
            "interpolation_factor",
        ]:
            logger.warning("Unknown keyword argument: {0} (1)", option, kwargs[option])
    resampled_period = (
        kwargs["resampled_period"] if "resampled_period" in options else None
    )
    if algorithm == "cc":
        method = kwargs["method"] if "method" in options else None
    elif algorithm == "cnw":
        interpolation_factor = (
            kwargs["interpolation_factor"]
            if "interpolation_factor" in options
            else None
        )

    past_sequences = 0 if period_history is None else len(period_history)
    logger.success(
        "Processing new sequence (Period: {0}; Drift: {1}; {2} past sequences).",
        this_period,
        this_drift,
        past_sequences,
    )
    #logger.debug(this_sequence[:, 0, 0])

    # Initialise sequences if empty
    # We assume if the sequence_history is empty, we should reset all the histories
    if sequence_history is None:
        logger.warning("Creating new histories.")
        sequence_history = []
        period_history = []
        drift_history = []
        shift_history = []
        global_solution = None
    elif not (
        len(period_history) == len(sequence_history)
        and len(drift_history) == len(sequence_history)
    ):
        # FIXME Doesn't catch shift_history or global_solution issues
        logger.critical("Histories have fallen out of sync! Resetting.")
        logger.critical(
            "{0},{1},{2},{3}",
            len(sequence_history),
            len(period_history),
            len(drift_history),
            len(shift_history),
        )
        sequence_history = []
        period_history = []
        drift_history = []
        shift_history = []
        global_solution = None

    if algorithm == "cc" and resampled_period is None:
        logger.info("Setting resampled period length to 80 (not provided by user).")
        resampled_period = 80

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
            # there are two other return statements in this function
            return (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                None,
                -1000,
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
            # there are two other return statements in this function
            return (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                None,
                -2000,
            )

    # Add latest reference frames to our sequence set
    if algorithm == "cc" or resampled_period is not None:
        # Resample this image sequence to have number of frames equal to the shared base
        this_resampled_sequence = hlp.interpolate_image_sequence(
            this_sequence, this_period, resampled_period / this_period, dtype="float"
        )[
            :resampled_period
        ]  # we use float to prevent any dtype-based overflow issues
        logger.debug(
            "{0} {1}", this_resampled_sequence.dtype, this_resampled_sequence.shape
        )
        sequence_history.append(this_resampled_sequence)
        period_history.append(this_period)
    elif algorithm == "cnw":
        # DO NOT convert target to a shared base as this should not be necessary for this algorithm
        sequence_history.append(this_sequence)
        period_history.append(this_period)

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
            "No drift correction is being applied. This will seriously impact phase locking."
        )

    # Print which algorithm, just for info
    if algorithm == "cc":
        logger.info("Using cross correlation method.")
    elif algorithm == "cnw":
        logger.info("Using cascading Needleman Wunsch algorithm.")

    # Update our shifts array.
    # Compare the current sequence with recent previous ones
    if len(sequence_history) > 1:
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(sequence_history) - max_offset - 1)
        for i in range(firstOne, len(sequence_history) - 1):
            logger.info(
                "Comparing reference period {0} and {1}", i, len(sequence_history) - 1
            )
            if algorithm == "cc":
                # Pairwise cross correlation between the new resampled period and historical (also resampled) periods
                # Returning the minimum cross-correlation for each pair (roll_factor) and the cross-correlation 'score'
                if this_drift is None:
                    (roll_factor, score) = cc.rolling_cross_correlation(
                        sequence_history[i],
                        sequence_history[-1],
                        resampled_period,
                        resampled_period,
                        resampled_period=None,
                        method=method,
                    )
                else:
                    logger.info("Drift given; accounting for drift.")
                    # FIXME - I think I need the following?
                    drift = [
                        drift_history[i][0] - drift_history[ref_seq_id][0],
                        drift_history[i][1] - drift_history[ref_seq_id][1],
                    ]
                    seq1, seq2 = hlp.drift_correction(
                        sequence_history[i], sequence_history[-1], this_drift
                    )
                    (roll_factor, score) = cc.rolling_cross_correlation(
                        seq1,
                        seq2,
                        resampled_period,
                        resampled_period,
                        resampled_period=None,
                        method=method,
                    )
                shift_history.append(
                    (
                        i,
                        len(sequence_history) - 1,
                        (roll_factor) % resampled_period,
                        score,
                    )
                )
                logger.info("Adding shift: {0}", shift_history[-1])
            elif algorithm == "cnw":
                # Cascading Needleman Wunsch alignment between the new, resampled period and historical periods
                # The target (ref_seq_phase) must be followed throughout these pairwise comparisons for accurate alignment; we use the global solution from the last round to always use the most 'corrected' target we can
                if i == ref_seq_id:
                    # If the comparison period is the one we have been given a target for just use ref_seq_phase
                    target = ref_seq_phase
                elif global_solution is not None:
                    # If the comparison period is not the one we have been given a target for then use the target inside the last global_solution
                    target = global_solution[i] % period_history[i]
                else:
                    # If there is no global_solution given, what should we do?
                    logger.critical("Erm! Crud.")
                logger.info(
                    "Using phase of {0} from reference sequence {1}", target, i,
                )

                if this_drift is None:
                    logger.debug("Calculate shift with no drift and cnw.")
                    (roll_factor, score) = cnw.cascading_needleman_wunsch(
                        sequence_history[i],
                        sequence_history[-1],
                        period_history[i],
                        period_history[-1],
                        interpolation_factor=interpolation_factor,
                        ref_seq_phase=target,
                    )
                else:
                    logger.info("Drift given; accounting for drift.")
                    # FIXME - I think I need the following?
                    drift = [
                        drift_history[i][0] - drift_history[ref_seq_id][0],
                        drift_history[i][1] - drift_history[ref_seq_id][1],
                    ]
                    seq1, seq2 = hlp.drift_correction(
                        sequence_history[i], sequence_history[-1], this_drift
                    )
                    logger.debug("** Calculate shift with drift and cnw.")
                    (roll_factor, score) = cnw.cascading_needleman_wunsch(
                        seq1,
                        seq2,
                        period_history[i],
                        period_history[-1],
                        interpolation_factor=interpolation_factor,
                        ref_seq_phase=target,
                    )
                shift_history.append((i, len(sequence_history) - 1, roll_factor, score))
                logger.info("Adding shift: {0}", shift_history[-1])

    logger.trace("Printing shifts:")
    logger.trace(shift_history)

    if algorithm == "cc":
        # Use multi-pass regression to use historical pairwise comparisons to minimise errors in the latest roll factor and get a global_solution
        global_solution = mr.make_shifts_self_consistent(
            shift_history, len(sequence_history), resampled_period
        )
    elif algorithm == "cnw":
        # Use multi-pass regression to use historical pairwise comparisons to minimise errors in the latest roll factor and get a global_solution
        global_solution = mr.make_shifts_self_consistent(
            shift_history,
            len(sequence_history),
            period_history,
            ref_seq_id=ref_seq_id,
            ref_seq_phase=ref_seq_phase,
        )

    logger.debug("Solution: {0}", global_solution)

    logger.debug("Reference Frame rolling by: {0}", global_solution[-1])

    if algorithm == "cc":
        # Convert initial target to the shared base too
        target = resampled_period * ref_seq_phase / period_history[ref_seq_id]
        logger.debug(
            "Target (base this_period {0}): {1}; Target (base resampled_period {2}): {3}",
            period_history[ref_seq_id],
            ref_seq_phase,
            resampled_period,
            target,
        )

        # Use the global_solution (modulo the shared base) and the initial target to identify  new target frame
        this_target = (
            target + (global_solution[-1] % resampled_period)
        ) % resampled_period
        logger.info(
            "Roll factor: {0}; New target (base resampled_sequence): {1}",
            global_solution[-1],
            this_target,
        )

        # Convert the new target frame into the original period
        this_target = this_period * this_target / resampled_period
    elif algorithm == "cnw":
        this_target = global_solution[-1] % this_period

    logger.success("New target (base this_period): {0}", this_target)

    # Note for developers:
    # there are two other return statements in this function
    return (
        sequence_history,
        period_history,
        drift_history,
        shift_history,
        global_solution,
        this_target,
    )
