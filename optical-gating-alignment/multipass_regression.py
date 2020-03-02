"""Modulo 2Pi multipass linear regression as used for phase locking in adaptive prospective optical gating systems."""


import numpy as np
from loguru import logger

# Set-up logger
logger.disable("optical-gating-alignment")

# Written by Jonathan Taylor, University of Glasgow
# jonathan.taylor@glasgow.ac.uk


def solve_for_shifts(shifts, number_of_sequences, ref_seq_id, ref_seq_phase):
    # Build a matrix/vector describing the system of equations Mx = a
    # This expects an input of 'shifts' consisting of triplets of (seq1Index, seq2Index, shift)
    # and an integer giving the number of sequences
    # (maximum value appearing for sequence index should be number_of_sequences-1)
    # Note that this function forces the absolute phase of the first sequence
    # to be equal to phaseForFirstSequence.
    M = np.zeros((len(shifts) + 1, number_of_sequences))
    a = np.zeros(len(shifts) + 1)
    w = a.copy()
    for (n, (i, j, shift, score)) in enumerate(shifts):
        # (i, j, shift, score) = shifts[n]
        M[n, i] = -1
        M[n, j] = 1
        a[n] = shift
        w[n] = 1.0 / score
    M[len(shifts), ref_seq_id] = 1
    a[len(shifts)] = ref_seq_phase
    w[len(shifts)] = 1

    # This weighted least squares is from http://stackoverflow.com/questions/19624997/understanding-scipys-least-square-function-with-irls
    Mw = M * np.sqrt(w[:, np.newaxis])
    aw = a * np.sqrt(w)
    (self_consistent_shifts, residuals, _, _) = np.linalg.lstsq(Mw, aw)
    return (self_consistent_shifts, residuals)


def solve_with_maximum_range(
    shifts, number_of_sequences, maximum_range, ref_seq_id, ref_seq_phase
):
    shifts_to_use = []
    for shift in shifts:
        # (i, j, shift, score) = shifts[n]
        if shift[1] <= shift[0] + maximum_range:
            shifts_to_use.append(shift)
    logger.info(
        "Solving using {0} of {1} constraints (max range {2})",
        len(shifts_to_use),
        len(shifts),
        maximum_range,
    )
    return solve_for_shifts(
        shifts_to_use, number_of_sequences, ref_seq_id, ref_seq_phase
    )


def adjust_shifts_to_match_solution(shifts, partial_solution, periods, warn_to=65536):
    # Now adjust the longer-distance shifts so they match our initial solution
    adjusted_shifts = []
    # DEVNOTE: JT's original code has isinstance (periods, (int,long))
    # This is Python2 syntac and not needed in Python3
    if isinstance(periods, int) or len(periods) == 1:
        period = periods
    for (i, j, shift, score) in shifts:
        # (i, j, shift, score) = shifts[n]
        if type(periods) is list and len(periods) > 1:
            period = periods[i]
        expected_wrapped_shift = (
            partial_solution[j] - partial_solution[i]) % period
        period_part = (
            partial_solution[j] - partial_solution[i]
        ) - expected_wrapped_shift
        discrepancy = expected_wrapped_shift - shift
        if abs(discrepancy) < (period / 4.0):
            # If discrepancy is small (positive or negative)
            # then add an appropriate number of periods to make it work
            adjusted_shift = shift + period_part
        elif abs(discrepancy) > (3 * period / 4.0):
            # Values look consistent, but cross a phase boundary
            if expected_wrapped_shift < shift:
                adjusted_shift = shift + (period_part - period)
            else:
                adjusted_shift = shift + (period_part + period)
        else:
            if j - i <= warn_to:
                logger.warning(
                    "Major discrepancy between approx expected value {0} and actual value {1} for {2} (distance {3}; score {4})",
                    expected_wrapped_shift,
                    shift,
                    (i, j),
                    j - i,
                    score,
                )
            # Exclude this shift because we aren't sure how to adjust it (yet)
            # Hopefully things may become clearer as we refine our estimated overall solution
            adjusted_shift = None

        if adjusted_shift is not None:
            adjusted_shifts.append((i, j, adjusted_shift, score))
    return adjusted_shifts


def make_shifts_self_consistent(
    shifts, number_of_sequences, period, ref_seq_id=0, ref_seq_phase=0
):
    # Given a set of what we think are the optimum relative time-shifts between different sequences
    # (both adjacent sequences, and some that are further apart), work out a global self-consistent solution.
    # The longer jumps serve to protect against gradual accumulation of random error in the absolute global phase,
    # which would creep in if we only ever considered the relative shifts of adjacent sequences.

    # First solve just using the shifts between adjacent slices (no phase wrapping)
    # TODO: add a more comprehensive comment here explaining the modulo-2pi issues that make the
    # shift problem a little bit awkward.
    (adjacent_shift_solution, adjacent_residuals) = solve_with_maximum_range(
        shifts, number_of_sequences, 1, ref_seq_id, ref_seq_phase
    )
    # Adjust the longer shifts to be consistent with the adjacent shift values
    # Don't warn about long-distance discrepancies, because those are fairly inevitable initially
    adjacent_shifts = adjust_shifts_to_match_solution(
        shifts, adjacent_shift_solution, period, warn_to=64
    )

    logger.info("Done first stage")

    # Now look for a solution that satisfies longer-range shifts as well.
    # If necessary, we could make a new adjustment of the shifts and repeat.
    # On a subsequent iteration we would have an improved estimate that might help us
    # decide which way to adjust long-range shifts that were initially unclear
    adjusted_shifts = list(adjacent_shifts)
    for r in [32, 128, 512, 2048]:
        (shift_solution, residuals) = solve_with_maximum_range(
            adjusted_shifts, number_of_sequences, r, ref_seq_id, ref_seq_phase
        )
        adjusted_shifts = adjust_shifts_to_match_solution(
            shifts, shift_solution, period
        )

    return (
        shift_solution,
        adjusted_shifts,
        adjacent_shift_solution,
        residuals,
        adjacent_residuals,
    )
