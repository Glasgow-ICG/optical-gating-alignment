import optical_gating_alignment.multipass_regression as mr
import numpy as np


def toy_shifts(knowledge_type="known"):
    shifts = []
    for i in np.arange(10):
        for j in np.arange(i + 1):
            if knowledge_type == "random":
                # assign a random shift in [0,2pi)
                # and a score of 1
                shifts.append((j, i, np.random.random() * 2 * np.pi, 1))
            elif knowledge_type == "known":
                # assign a deterministic shift in [0,2pi)
                # and a score of 1
                if j > 0 or i > 0:
                    shifts.append((j, i, shifts[-1][2] + (i - j) / 10 * 2 * np.pi, 1))
                else:
                    shifts.append((j, i, (i - j) / 10 * 2 * np.pi, 1))

    number_of_sequences = 10
    return shifts, number_of_sequences


def test_solve_for_shifts():
    # assumes the code was correct at the time this test was made
    # TODO do something more deterministic
    # toy shifts
    shifts, number_of_sequences = toy_shifts(knowledge_type="known")
    ref_seq_id = 0  # TODO vary
    ref_seq_phase = 0  # TODO vary

    known_shifts = [
        5.79986336,
        4.44216807,
        3.58537008,
        4.38504821,
        7.75512033,
        14.89510363,
        27.29011465,
        46.71086923,
        75.21368258,
        115.14046921,
    ]

    self_consistent_shifts = mr.solve_for_shifts(
        shifts, number_of_sequences, ref_seq_id, ref_seq_phase
    )

    print(self_consistent_shifts)

    assert np.all(np.abs(self_consistent_shifts - known_shifts) < 1e-6)


def test_solve_with_maximum_range():
    # toy shifts
    shifts, number_of_sequences = toy_shifts(knowledge_type="known")
    ref_seq_id = 0  # TODO vary
    ref_seq_phase = 0  # TODO vary

    accurate = []
    for maximum_range in np.arange(1, 10):
        self_consistent_shifts = mr.solve_with_maximum_range(
            shifts, number_of_sequences, maximum_range, ref_seq_id, ref_seq_phase
        )
        if maximum_range == 1:
            previous_shifts = self_consistent_shifts.copy()
        else:
            previous_shifts = self_consistent_shifts.copy()
            accurate.append(
                np.all(np.abs(self_consistent_shifts - previous_shifts) < 1e6)
            )

    assert np.all(accurate)


def test_adjust_shifts_to_match_solution_period():
    # assumes the code was correct at the time this test was made
    # TODO do something more deterministic
    # toy shifts
    shifts, number_of_sequences = toy_shifts(knowledge_type="known")
    ref_seq_id = 0  # TODO vary
    ref_seq_phase = 0  # TODO vary
    maximum_range = 3  # TODO vary?
    period = 80  # TODO vary?

    known_adjusted_shifts = [
        0.0,
        0.62831853,
        0.62831853,
        1.88495559,
        2.51327412,
        2.51327412,
        4.39822972,
        5.65486678,
        6.28318531,
        6.28318531,
        8.79645943,
        10.68141502,
        11.93805208,
        12.56637061,
        12.56637061,
        15.70796327,
        18.22123739,
        20.10619298,
        21.36283004,
        21.99114858,
        25.76105976,
        28.90265241,
        31.41592654,
        33.30088213,
        34.55751919,
        35.18583772,
        39.58406744,
        43.35397862,
        46.49557127,
        49.0088454,
        50.89380099,
        52.15043805,
        69.11503838,
        71.6283125,
        73.51326809,
        74.76990516,
        -4.60177631,
        161.05309046,
        166.07963871,
        101.78760198,
        103.04423904,
        23.67255757,
    ]
    self_consistent_shifts = mr.solve_with_maximum_range(
        shifts, number_of_sequences, maximum_range, ref_seq_id, ref_seq_phase
    )
    adjusted_shifts = mr.adjust_shifts_to_match_solution(
        shifts, self_consistent_shifts, period
    )

    adjusted_shifts = np.stack(adjusted_shifts)[:, 2]
    print(adjusted_shifts)

    assert np.all(np.abs(known_adjusted_shifts - adjusted_shifts) < 1e-6)


def test_adjust_shifts_to_match_solution_periods():
    # assumes the code was correct at the time this test was made
    # TODO do something more deterministic
    # toy shifts
    shifts, number_of_sequences = toy_shifts(knowledge_type="known")
    ref_seq_id = 0  # TODO vary
    ref_seq_phase = 0  # TODO vary
    maximum_range = 3  # TODO vary?
    periods = np.repeat([80], number_of_sequences)  # TODO vary?

    known_adjusted_shifts = [
        0.0,
        0.62831853,
        0.62831853,
        1.88495559,
        2.51327412,
        2.51327412,
        4.39822972,
        5.65486678,
        6.28318531,
        6.28318531,
        8.79645943,
        10.68141502,
        11.93805208,
        12.56637061,
        12.56637061,
        15.70796327,
        18.22123739,
        20.10619298,
        21.36283004,
        21.99114858,
        25.76105976,
        28.90265241,
        31.41592654,
        33.30088213,
        34.55751919,
        35.18583772,
        39.58406744,
        43.35397862,
        46.49557127,
        49.0088454,
        50.89380099,
        52.15043805,
        69.11503838,
        71.6283125,
        73.51326809,
        74.76990516,
        -4.60177631,
        161.05309046,
        166.07963871,
        101.78760198,
        103.04423904,
        23.67255757,
    ]
    self_consistent_shifts = mr.solve_with_maximum_range(
        shifts, number_of_sequences, maximum_range, ref_seq_id, ref_seq_phase
    )
    adjusted_shifts = mr.adjust_shifts_to_match_solution(
        shifts, self_consistent_shifts, periods
    )

    adjusted_shifts = np.stack(adjusted_shifts)[:, 2]
    print(adjusted_shifts)

    assert np.all(np.abs(known_adjusted_shifts - adjusted_shifts) < 1e-6)


def test_make_shifts_self_consistent():
    # assumes the code was correct at the time this test was made
    # TODO do something more deterministic
    # toy shifts
    shifts, number_of_sequences = toy_shifts(knowledge_type="known")
    ref_seq_id = 0  # TODO vary
    ref_seq_phase = 0  # TODO vary
    periods = np.repeat([80], number_of_sequences)  # TODO vary?

    known_shift_solution = [
        -10.03461273,
        -12.83180584,
        -13.68860384,
        -12.88892571,
        -9.51885359,
        7.68531411,
        18.99504766,
        37.42719194,
        42.48591781,
        67.57342646,
    ]

    shift_solution = mr.make_shifts_self_consistent(
        shifts,
        number_of_sequences,
        periods,
        ref_seq_id=ref_seq_id,
        ref_seq_phase=ref_seq_phase,
    )

    print(shift_solution)
    assert np.all(np.abs(known_shift_solution - shift_solution) < 1e-6)
