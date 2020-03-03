import optical_gating_alignment as oga
import optical_gating_alignment.helper as hlp
import optical_gating_alignment.cross_correlation as cc
import optical_gating_alignment.cascading_needleman_wunsch as cnw
import numpy as np
import pytest


def test_version():
    assert oga.__version__ == "2.0.0"


def toy_sequence(length=0, seq_type="image", knowledge_type="random", dtype="uint8"):
    if seq_type == "image":
        if knowledge_type == "random":
            sequence = np.random.randint(0, 2 ** 8, [length, 64, 128]).astype(dtype)
        if knowledge_type == "known":
            # 'string' with uint8 triangular intensity pattern
            string = [0, 3, 15, 63, 255, 63, 15, 3, 0]
            # convert to uint8 'image sequence' (1x1 frame)
            sequence = np.asarray(string, "uint8").reshape([len(string), 1, 1])
            # convert to rectangular array (64x128 frame)
            sequence = np.repeat(np.repeat(sequence, 64, 1), 128, 2).astype(dtype)
    if seq_type == "alignment":
        sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return sequence


def test_drift_correction_uint8():
    # create a rectangular uint8 'image' sequence
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )

    # create a rolled, i.e. drifted, version
    # do four directions (+,+), (+,-), (-,+), (-,-)
    drift_x = np.random.randint(1, 5)
    drift_y = np.random.randint(1, 5)
    for x_dir in [+1, -1]:
        for y_dir in [+1, -1]:
            sequence2 = np.roll(sequence1, x_dir * drift_x, axis=1)
            sequence2 = np.roll(sequence2, y_dir * drift_y, axis=2)

            # correct for positive-positive drift
            corrected1, corrected2 = hlp.drift_correction(
                sequence1, sequence2, [x_dir * drift_x, y_dir * drift_y]
            )

            assert np.all(corrected1[0] == corrected2[0])


def test_drift_correction_uint16():
    # create a rectangular uint8 'image' 'sequence'
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint16"
    )

    # create a rolled, i.e. drifted, version
    drift_x = np.random.randint(1, 5)
    drift_y = np.random.randint(1, 5)
    sequence2 = np.roll(sequence1, drift_x, axis=1)
    sequence2 = np.roll(sequence2, drift_y, axis=2)

    # correct for drift
    corrected1, corrected2 = hlp.drift_correction(
        sequence1, sequence2, [drift_x, drift_y]
    )

    assert np.all(corrected1[0] == corrected2[0])


def test_resample_sequence_uint8_not1or2():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use non-integer period
    current_period = period_int - np.random.rand(1)

    # determine resampling
    resample_factor = np.random.randint(3, 5)
    new_period = int(period_int * resample_factor)

    # resample with random factor
    resampled_sequence = hlp.resample_sequence(sequence, current_period, new_period)

    assert len(resampled_sequence) == new_period


def test_resample_sequence_uint8_2():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(
        sequence, len(sequence), 2 * len(sequence)
    )

    assert np.all(resampled_sequence[::2, ...] == sequence)


def test_resample_sequence_uint8_1():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(sequence, len(sequence), len(sequence))

    assert np.all(resampled_sequence == sequence)


def test_resample_sequence_uint16_not1or2():
    # create a rectangular uint16 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    # use non-integer period
    current_period = period_int - np.random.rand(1)

    # determine resampling
    resample_factor = np.random.randint(3, 5)
    new_period = int(period_int * resample_factor)

    # resample with random factor
    resampled_sequence = hlp.resample_sequence(sequence, current_period, new_period)

    assert len(resampled_sequence) == new_period


def test_resample_sequence_uint16_2():
    # create a rectangular uint16 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(
        sequence, len(sequence), 2 * len(sequence)
    )

    assert np.all(resampled_sequence[::2, ...] == sequence)


def test_resample_sequence_uint16_1():
    # create a rectangular uint16 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(sequence, len(sequence), len(sequence))

    assert np.all(resampled_sequence == sequence)


def test_resample_sequence_uint8_known():
    # create a rectangular uint8 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.resample_sequence(sequence, period, 80)

    # this was very manual
    expected = [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        15,
        19,
        24,
        29,
        34,
        39,
        43,
        48,
        53,
        58,
        63,
        82,
        101,
        120,
        139,
        159,
        178,
        197,
        216,
        235,
        255,
        235,
        216,
        197,
        178,
        159,
        139,
        120,
        101,
        82,
        63,
        58,
        53,
        48,
        43,
        39,
        34,
        29,
        24,
        19,
        15,
        13,
        12,
        11,
        10,
        9,
        7,
        6,
        5,
        4,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
    ]

    assert (
        np.all(expected == resampled_sequence[:, 1, 1])
        and resampled_sequence.dtype == np.uint8
    )


def test_resample_sequence_uint16_known():
    # create a rectangular uint16 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.resample_sequence(sequence, period, 80)

    # this was very manual
    expected = [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        15,
        19,
        24,
        29,
        34,
        39,
        43,
        48,
        53,
        58,
        63,
        82,
        101,
        120,
        139,
        159,
        178,
        197,
        216,
        235,
        255,
        235,
        216,
        197,
        178,
        159,
        139,
        120,
        101,
        82,
        63,
        58,
        53,
        48,
        43,
        39,
        34,
        29,
        24,
        19,
        15,
        13,
        12,
        11,
        10,
        9,
        7,
        6,
        5,
        4,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
    ]

    assert (
        np.all(expected == resampled_sequence[:, 1, 1])
        and resampled_sequence.dtype == np.uint16
    )


def test_cross_correlation_uint8_self():
    # create a rectangular uint8 'image' sequence where each image is the same
    sequence = toy_sequence(
        length=1, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    sequence = np.tile(sequence, [10, 1, 1])

    # flatten each frame for 2D cross correlation
    sequence_flat = sequence.reshape([len(sequence), -1])

    # determine score with self
    scores = cc.cross_correlation(sequence_flat, sequence_flat)

    # this does not equal zero, which I think would make sense?

    assert np.all(scores == scores[0])


def test_cross_correlation_uint8_notself():
    # create a rectangular uint8 'image' sequence where each image is different
    sequence = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )

    # flatten each frame for 2D cross correlation
    sequence_flat = sequence.reshape([len(sequence), -1])

    # determine score with anti-self
    scores = cc.cross_correlation(sequence_flat, sequence_flat)

    assert np.all(scores[1:] != scores[0])


def test_cross_correlation_uint16_self():
    # create a rectangular uint16 'image' sequence where each image is the same
    sequence = toy_sequence(
        length=1, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    sequence = np.tile(sequence, [10, 1, 1])

    # flatten each frame for 2D cross correlation
    sequence_flat = sequence.reshape([len(sequence), -1])

    # determine score with self
    scores = cc.cross_correlation(sequence_flat, sequence_flat)

    # this does not equal zero, which I think would make sense?

    assert np.all(scores == scores[0])


def test_cross_correlation_uint16_notself():
    # create a rectangular uint16 'image' sequence where each image is different
    sequence = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint16"
    )

    # flatten each frame for 2D cross correlation
    sequence_flat = sequence.reshape([len(sequence), -1])

    # determine score with anti-self
    scores = cc.cross_correlation(sequence_flat, sequence_flat)

    assert np.all(scores[1:] != scores[0])


def test_v_minimum_simple():
    # very simple example
    y1 = 1.0
    y2 = 0.1
    y3 = 1.0
    x, y = cc.v_minimum(y1, y2, y3)

    assert x == 0.0 and y == 0.1


def test_v_minimum_complex():
    # slightly more complex example
    y1 = 2.5
    y2 = 0.1
    y3 = 1.0
    x, y = cc.v_minimum(y1, y2, y3)

    assert x == 0.3125 and y == -0.65


def test_v_minimum_flipped():
    # slightly more complex example - flipped
    y1 = 2.5
    y2 = 0.1
    y3 = 1.0
    x, y = cc.v_minimum(y3, y2, y1)

    assert x == -0.3125 and y == -0.65


def test_minimum_score_simple():
    # very simple example
    scores = [1.0, 0.1, 1.0]
    minimum_position, minimum_value = cc.minimum_score(scores)

    # the 1.0+ is because this function returns position from the start
    assert minimum_position == 1.0 + 0.0 and minimum_value == 0.1


def test_minimum_score_complex():
    # slightly more complex example
    scores = [2.5, 0.1, 1.0]
    minimum_position, minimum_value = cc.minimum_score(scores)

    assert minimum_position == 1.0 + 0.3125 and minimum_value == -0.65


def test_minimum_score_flipped():
    # slightly more complex example - flipped
    scores = [1.0, 0.1, 2.5]
    minimum_position, minimum_value = cc.minimum_score(scores)

    assert minimum_position == 1.0 - 0.3125 and minimum_value == -0.65


def test_rolling_cross_correlation_uint8_equal_start():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = 0 % period1
    sequence2 = np.roll(sequence1, roll, axis=0)
    period2 = period1

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint8_equal_end():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by period)
    roll = len(sequence1) % period1
    sequence2 = np.roll(sequence1, roll, axis=0)
    period2 = period1

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint8_equal_rand():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    # shifted string (never rolls by zero or period)
    roll = np.random.randint(1, len(sequence1) - 1) % period1
    sequence2 = np.roll(sequence1, roll, axis=0)
    period2 = period1

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point errors and ambiguity
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-2


def test_rolling_cross_correlation_uint16_equal_start():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = 0 % period1
    sequence2 = np.roll(sequence1, roll, axis=0)
    period2 = period1

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint16_equal_end():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by period)
    roll = len(sequence1) % period1
    sequence2 = np.roll(sequence1, roll, axis=0)
    period2 = period1

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint16_equal_rand():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    # shifted string (never rolls by zero or period)
    roll = np.random.randint(1, len(sequence1) - 1) % period1
    sequence2 = np.roll(sequence1, roll, axis=0)
    period2 = period1

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point errors and ambiguity
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-2


def test_rolling_cross_correlation_uint8_notequal_start():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = 0 % period1
    sequence2 = np.roll(sequence1[:-1], roll, axis=0)
    period2 = len(sequence2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint8_notequal_end():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = len(sequence1) % period1
    sequence2 = np.roll(sequence1[:-1], roll, axis=0)
    period2 = len(sequence2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint8_notequal_rand():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = np.random.randint(1, len(sequence1) - 1) % period1
    sequence2 = np.roll(sequence1[:-1], roll, axis=0)
    period2 = len(sequence2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - roll - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint16_notequal_start():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = 0 % period1
    sequence2 = np.roll(sequence1[:-1], roll, axis=0)
    period2 = len(sequence2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint16_notequal_end():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = len(sequence1) % period1
    sequence2 = np.roll(sequence1[:-1], roll, axis=0)
    period2 = len(sequence2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint16_notequal_rand():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    # shifted string (by zero)
    roll = np.random.randint(1, len(sequence1) - 1) % period1
    sequence2 = np.roll(sequence1[:-1], roll, axis=0)
    period2 = len(sequence2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - roll - 0.4469 < 1e-3


# FIXME roll factor seems to be out by phase1 - will do after I work out WTF I wrote
def test_get_roll_factor_at():
    # toy alignment data
    alignment1 = toy_sequence(seq_type="alignment")
    period1 = 10  # use integer
    # shifted alignment (by zero)
    roll = 0
    alignment2 = np.roll(alignment1, roll)

    roll_factor = []
    for phase1 in np.arange(period1):
        # get roll factor
        roll_factor.append(cnw.get_roll_factor_at(alignment1, alignment2, phase1))

    assert np.all(roll_factor == roll_factor[0])
