import optical_gating_alignment.cascading_needleman_wunsch as cnw
import numpy as np
import j_py_sad_correlation as jps
from tests.test_helper import toy_sequence


def test_get_roll_factor_at_list():
    # to test that the function correctly handles ilists (when given)
    # toy alignment data
    alignment1 = list(toy_sequence(seq_type="alignment"))

    assert cnw.get_roll_factor_at(alignment1, alignment1, 0) == 0


def test_get_roll_factor_at_lower_gap():
    # to test that the function correctly handles gaps in the lower bound
    # toy alignment data
    alignment1 = list(toy_sequence(seq_type="alignment"))
    alignment1.insert(2, -1)

    assert cnw.get_roll_factor_at(alignment1, alignment1, 1.5) == 0


def test_get_roll_factor_at_upper_gap():
    # to test that the function correctly handles gaps in the upper bound
    # toy alignment data
    alignment1 = list(toy_sequence(seq_type="alignment"))
    alignment1.insert(3, -1)

    assert cnw.get_roll_factor_at(alignment1, alignment1, 1.5) == 0


def test_get_roll_factor_at():
    # toy alignment data
    alignment1 = toy_sequence(seq_type="alignment")

    accurate = []
    for roll in np.arange(len(alignment1) + 1):
        # shift alignment
        alignment2 = np.roll(alignment1, roll, axis=0)

        for phase1 in np.arange(0, len(alignment1), 0.5):
            # get roll factor
            roll_factor = cnw.get_roll_factor_at(alignment1, alignment2, phase1)
            print(roll, phase1, roll_factor, np.abs(roll_factor - roll) < 0.1)

            # small catch for rounding errors
            accurate.append(np.abs(roll_factor - (roll % len(alignment1))) < 0.1)

    assert np.all(accurate)


def test_get_roll_factor_at_wrap1():
    # toy alignment data
    alignment1 = toy_sequence(seq_type="alignment") + 1
    # the +1 makes a phase of 0 be at the wrap1 point

    accurate = []
    for roll in np.arange(len(alignment1) + 1):
        # shift alignment
        alignment2 = np.roll(alignment1, roll, axis=0)

        # get roll factor
        roll_factor = cnw.get_roll_factor_at(alignment1, alignment2, 0)

        # small catch for rounding errors and for wrap point induced errors
        accurate.append(np.abs(roll_factor - (roll % len(alignment1))) < 0.1)

    assert np.all(accurate)


def test_get_roll_factor_at_wrap2():
    # toy alignment data
    alignment1 = toy_sequence(seq_type="alignment")
    alignment1 = np.roll(alignment1, -2, axis=0)
    # the roll -2 makes there be a wrap point 2 for a period-0.5 phase

    accurate = []
    for roll in np.arange(len(alignment1) + 1):
        # shift alignment
        alignment2 = np.roll(alignment1, roll, axis=0)

        # get roll factor
        roll_factor = cnw.get_roll_factor_at(
            alignment1, alignment2, len(alignment1) - 0.5
        )

        print(roll, 9.5, roll_factor)

        # small catch for rounding errors and for wrap point induced errors
        accurate.append(np.abs(roll_factor - (roll % len(alignment1))) < 0.1)

    assert np.all(accurate)


def test_fill_traceback_matrix_self_uint8():
    # assumes the code was correct at the time this test was made
    # toy image sequences
    sequence1 = toy_sequence(seq_type="image", knowledge_type="tiny", dtype="uint8")

    # score matrix
    score_matrix = jps.sad_grid(sequence1, sequence1)

    # traceback matrix
    traceback_matrix = cnw.fill_traceback_matrix(score_matrix)

    known_traceback = [
        [0, -8192, -507904, -2580480, -3088384, -3088384,],
        [-8192, 0, -507904, -2588672, -3096576, -3096576,],
        [-507904, -507904, 0, -1572864, -1581056, -2080768,],
        [-2580480, -2588672, -1572864, 0, -1564672, -3637248,],
        [-3088384, -3096576, -1581056, -1564672, 0, -507904,],
        [-3088384, -3096576, -2080768, -3637248, -507904, 0],
    ]

    assert np.all(traceback_matrix == known_traceback)


# def test_fill_traceback_matrix_self_uint16():
#     # assumes the code was correct at the time this test was made
#     # toy image sequences
#     sequence1 = toy_sequence(seq_type="image", knowledge_type="tiny", dtype="uint16")

#     # score matrix
#     score_matrix = jps.sad_grid(sequence1, sequence1)

#     # traceback matrix
#     traceback_matrix = cnw.fill_traceback_matrix(score_matrix)

#     known_traceback = [
#         [0, -8192, -507904, -2580480, -3088384, -3088384,],
#         [-8192, 0, -507904, -2588672, -3096576, -3096576,],
#         [-507904, -507904, 0, -1572864, -1581056, -2080768,],
#         [-2580480, -2588672, -1572864, 0, -1564672, -3637248,],
#         [-3088384, -3096576, -1581056, -1564672, 0, -507904,],
#         [-3088384, -3096576, -2080768, -3637248, -507904, 0],
#     ]

#     assert np.all(traceback_matrix == known_traceback)


def test_traverse_traceback_matrix_list():
    # toy image sequences
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )

    # score matrix
    score_matrix = jps.sad_grid(sequence1, sequence1)

    # traceback matrix
    traceback_matrix = cnw.fill_traceback_matrix(score_matrix)

    # traverse matrix
    alignment1, alignment2 = cnw.traverse_traceback_matrix(
        list(sequence1), list(sequence1), traceback_matrix
    )

    # compare alignments (ignoring indels)
    alignment1 = alignment1[alignment1 != -1]  # remove indels
    alignment2 = alignment2[alignment2 != -1]  # remove indels

    assert np.all(alignment1 == alignment2)


def test_traverse_traceback_matrix_self_uint8():
    # toy image sequences
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )

    accurate = []
    for roll in np.arange(len(sequence1) + 1):
        sequence2 = np.roll(sequence1, roll, axis=0)

        # score matrix
        score_matrix = jps.sad_grid(sequence1, sequence2)

        # traceback matrix
        traceback_matrix = cnw.fill_traceback_matrix(score_matrix)

        # traverse matrix
        alignment1, alignment2 = cnw.traverse_traceback_matrix(
            sequence1, sequence2, traceback_matrix
        )

        # compare alignments (ignoring indels)
        alignment1 = alignment1[alignment1 != -1]  # remove indels
        alignment2 = alignment2[alignment2 != -1]  # remove indels
        accurate.append(alignment1 == alignment2)

    assert np.all(accurate)


# def test_traverse_traceback_matrix_self_uint16():
#     # toy image sequences
#     sequence1 = toy_sequence(
#         length=10, seq_type="image", knowledge_type="random", dtype="uint16"
#     )

#     accurate = []
#     for roll in np.arange(len(sequence1) + 1):
#         sequence2 = np.roll(sequence1, roll, axis=0)

#         # score matrix
#         score_matrix = jps.sad_grid(sequence1, sequence2)

#         # traceback matrix
#         traceback_matrix = cnw.fill_traceback_matrix(score_matrix)

#         # traverse matrix
#         alignment1, alignment2 = cnw.traverse_traceback_matrix(
#             sequence1, sequence2, traceback_matrix
#         )

#         # compare alignments (ignoring indels)
#         alignment1 = alignment1[alignment1 != -1]  # remove indels
#         alignment2 = alignment2[alignment2 != -1]  # remove indels
#         accurate.append(alignment1 == alignment2)

#     assert np.all(accurate)


def test_roll_score_matrix_self_uint8():
    # toy image sequences
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )

    # score matrix
    score_matrix = jps.sad_grid(sequence1, sequence1)

    accurate = []
    for roll in np.arange(sequence1.shape[0] + 1):
        rolled_score_matrix = cnw.roll_score_matrix(
            score_matrix, roll_factor=roll, axis=0
        )
        accurate.append(
            np.all(rolled_score_matrix == np.roll(score_matrix, roll, axis=0))
        )

    accurate = []
    for roll in np.arange(sequence1.shape[1] + 1):
        rolled_score_matrix = cnw.roll_score_matrix(
            score_matrix, roll_factor=roll, axis=1
        )
        accurate.append(
            np.all(rolled_score_matrix == np.roll(score_matrix, roll, axis=1))
        )

    assert np.all(accurate)


# def test_roll_score_matrix_self_uint16():
#     # toy image sequences
#     sequence1 = toy_sequence(
#         length=10, seq_type="image", knowledge_type="random", dtype="uint16"
#     )

#     # score matrix
#     score_matrix = jps.sad_grid(sequence1, sequence1)

#     accurate = []
#     for roll in np.arange(sequence1.shape[0] + 1):
#         rolled_score_matrix = cnw.roll_score_matrix(
#             score_matrix, roll_factor=roll, axis=0
#         )
#         accurate.append(
#             np.all(rolled_score_matrix == np.roll(score_matrix, roll, axis=0))
#         )

#     accurate = []
#     for roll in np.arange(sequence1.shape[1] + 1):
#         rolled_score_matrix = cnw.roll_score_matrix(
#             score_matrix, roll_factor=roll, axis=1
#         )
#         accurate.append(
#             np.all(rolled_score_matrix == np.roll(score_matrix, roll, axis=1))
#         )

#     assert np.all(accurate)


def test_construct_cascade_self_uint8():
    # assumes the code was correct at the time this test was made
    # toy image sequences
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="known", dtype="uint8"
    )

    # score matrix
    score_matrix = jps.sad_grid(sequence1, sequence1)

    # create cascade of traceback matrices
    cascade = cnw.construct_cascade(score_matrix)

    known_scores = [
        0,
        -8192,
        -32768,
        -335872,
        -2015232,
        -5160960,
        -1523712,
        -229376,
        -16384,
    ]

    print(cascade[-1, -1, :])

    assert np.all(cascade[-1, -1, :] == known_scores)


# def test_construct_cascade_self_uint16():
#     # assumes the code was correct at the time this test was made
#     # toy image sequences
#     sequence1 = toy_sequence(
#         length=10, seq_type="image", knowledge_type="known", dtype="uint16"
#     )

#     # score matrix
#     score_matrix = jps.sad_grid(sequence1, sequence1)

#     # create cascade of traceback matrices
#     cascade = cnw.construct_cascade(score_matrix)

#     known_scores = [
#         0,
#         -8192,
#         -32768,
#         -335872,
#         -2015232,
#         -5160960,
#         -1523712,
#         -229376,
#         -16384,
#     ]

#     assert np.all(cascade[-1, -1, :] == known_scores)


def test_wrap_and_roll_gapless():
    # this test uses a gapless sequence, therefore should be equivalent to np.roll
    alignment1Wrapped = toy_sequence(seq_type="alignment")
    period1 = 10  # use integer

    accurate = []
    for roll in np.arange(period1 + 1):
        alignment1 = cnw.wrap_and_roll(alignment1Wrapped, period1, roll)
        alignment2 = np.roll(alignment1Wrapped, roll, axis=0)
        accurate.append(np.all(alignment1 == alignment2))

    assert np.all(accurate)


def test_wrap_and_roll_gapped():
    # this test uses a gapped sequence
    # but removes the gaps before comparing to np.roll
    alignment1Wrapped = toy_sequence(seq_type="alignment")
    period1 = 10  # use integer

    accurate = []
    for gap in np.arange(period1 + 1):
        alignment1Gapped = np.insert(alignment1Wrapped, gap, -1)
        for roll in np.arange(period1 + 1):
            alignment1 = cnw.wrap_and_roll(alignment1Gapped, period1, roll)
            alignment1 = alignment1[alignment1 >= 0]
            alignment2 = np.roll(alignment1Wrapped, roll, axis=0)
            print(alignment1Gapped, alignment1, alignment2)
            accurate.append(np.all(alignment1 == alignment2))

    assert np.all(accurate)


def test_wrap_and_roll_doublegapped():
    # this test uses a double gapped sequence
    # but removes the gaps before comparing to np.roll
    alignment1Wrapped = toy_sequence(seq_type="alignment")
    period1 = 10  # use integer

    accurate = []
    for gap in np.arange(period1 + 1):
        alignment1Gapped = np.insert(alignment1Wrapped, gap, -1)
        alignment1Gapped = np.insert(alignment1Wrapped, gap, -1)
        for roll in np.arange(period1 + 1):
            alignment1 = cnw.wrap_and_roll(alignment1Gapped, period1, roll)
            alignment1 = alignment1[alignment1 >= 0]
            alignment2 = np.roll(alignment1Wrapped, roll, axis=0)
            print(alignment1Gapped, alignment1, alignment2)
            accurate.append(np.all(alignment1 == alignment2))

    assert np.all(accurate)


def test_cascading_needleman_wunsch_self_uint8():
    # toy image sequences
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="known", dtype="uint8"
    )

    accurate = []
    for roll in np.arange(len(sequence1) + 1):
        sequence2 = np.roll(sequence1, roll, axis=0)

        for phase in np.arange(0.5, len(sequence1), 0.5):
            alignmentA, alignmentB, roll_factor, score = cnw.cascading_needleman_wunsch(
                sequence1,
                sequence2,
                period=None,
                template_period=None,
                gap_penalty=0,
                interpolation_factor=None,
                ref_seq_phase=phase,
            )

            # small catch for rounding errors and for wrap point induced errors
            accurate.append(np.abs(roll_factor - (roll % len(sequence1))) < 0.1)
            print(roll, phase, roll_factor)

    assert np.all(accurate)


# TODO get 16 bit working with jps
# def test_cascading_needleman_wunsch_self_uint16():
#     # toy image sequences
#     sequence1 = toy_sequence(
#         length=10, seq_type="image", knowledge_type="known", dtype="uint16"
#     )

#     accurate = []
#     for roll in np.arange(len(sequence1) + 1):
#         sequence2 = np.roll(sequence1, roll, axis=0)

#         for phase in np.arange(0.5, len(sequence1), 0.5):
#             alignmentA, alignmentB, roll_factor, score = cnw.cascading_needleman_wunsch(
#                 sequence1,
#                 sequence2,
#                 period=None,
#                 template_period=None,
#                 gap_penalty=0,
#                 interpolation_factor=None,
#                 ref_seq_phase=phase,
#             )

#             # small catch for rounding errors and for wrap point induced errors
#             accurate.append(np.abs(roll_factor - (roll % len(sequence1))) < 0.1)
#             print(roll, phase, roll_factor)

#     assert np.all(accurate)


# TODO get interp working
# def test_cascading_needleman_wunsch_self_interp_uint8():
#     # toy image sequences
#     sequence1 = toy_sequence(
#         length=10, seq_type="image", knowledge_type="known", dtype="uint8"
#     )

#     accurate = []
#     for interp in [2]:  # np.arange(1, 2):
#         for roll in [1]:  # np.arange(len(sequence1) + 1):
#             sequence2 = np.roll(sequence1, roll, axis=0)

#             for phase in np.arange(0.5, len(sequence1), 0.5):
#                 (
#                     alignmentA,
#                     alignmentB,
#                     roll_factor,
#                     score,
#                 ) = cnw.cascading_needleman_wunsch(
#                     sequence1,
#                     sequence2,
#                     period=None,
#                     template_period=None,
#                     gap_penalty=0,
#                     interpolation_factor=interp,
#                     ref_seq_phase=phase,
#                 )

#                 # small catch for rounding errors and for wrap point induced errors
#                 accurate.append(np.abs(roll_factor - (roll % len(sequence1))) < 0.1)
#                 print(interp, roll, phase, roll_factor)

#     assert np.all(accurate)


# def test_cascading_needleman_wunsch_self_interp_uint16():
#     # this test could do with being more robust - but that might presume the current code is correct
#     # toy image sequences
#     sequence1 = toy_sequence(
#         length=10, seq_type="image", knowledge_type="known", dtype="uint16"
#     )
#     period1 = len(sequence1)  # use integer

#     accurate = []
#     for roll in np.arange(len(sequence1) + 1):
#         sequence2 = np.roll(sequence1, roll, axis=0)
#         period2 = len(sequence1)  # use integer

#         alignmentA, alignmentB, roll_factor, score = cnw.cascading_needleman_wunsch(
#             sequence1,
#             sequence2,
#             period=None,
#             template_period=None,
#             gap_penalty=0,
#             interpolation_factor=2.0,
#             ref_seq_phase=0,
#         )

#         # catch a range of roll factors, this deals with duplicate intensities and such
#         accurate.append(np.abs(roll_factor - roll) < 1)

#     assert np.all(accurate)
