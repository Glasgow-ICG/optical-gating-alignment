import optical_gating_alignment as oga
import optical_gating_alignment.helper as hlp
import optical_gating_alignment.cross_correlation as cc
import optical_gating_alignment.cascading_needleman_wunsch as cnw
import numpy as np
import pytest


def test_version():
    assert oga.__version__ == "2.0.0"


def test_drift_correction_uint8():
    # create a rectangular uint8 'image sequence'
    image1 = np.random.randint(0, 2 ** 8, [64, 128]).astype("uint8")
    sequence1 = image1[np.newaxis, ...]  # a sequence of length 1

    # create a rolled, i.e. drifted, version
    # do four directions (+,+), (+,-), (-,+), (-,-)
    drift_x = np.random.randint(1, 5)
    drift_y = np.random.randint(1, 5)
    for x_dir in [+1, -1]:
        for y_dir in [+1, -1]:
            image2 = np.roll(image1, x_dir * drift_x, axis=0)
            image2 = np.roll(image2, y_dir * drift_y, axis=1)
            sequence2 = image2[np.newaxis, ...]

            # correct for positive-positive drift
            corrected1, corrected2 = hlp.drift_correction(
                sequence1, sequence2, [x_dir * drift_x, y_dir * drift_y]
            )

            assert np.all(corrected1[0] == corrected2[0])


def test_drift_correction_uint16():
    # create a rectangular uint16 'image sequence'
    image1 = np.random.randint(0, 2 ** 8, [64, 128]).astype("uint16")
    sequence1 = image1[np.newaxis, ...]  # a sequence of length 1

    # create a rolled, i.e. drifted, version
    drift_x = np.random.randint(1, 5)
    drift_y = np.random.randint(1, 5)
    image2 = np.roll(image1, drift_x, axis=0)
    image2 = np.roll(image2, drift_y, axis=1)
    sequence2 = image2[np.newaxis, ...]

    # correct for drift
    corrected1, corrected2 = hlp.drift_correction(
        sequence1, sequence2, [drift_x, drift_y]
    )

    assert np.all(corrected1[0] == corrected2[0])


def test_resample_sequence_not1or2():
    # create a rectangular 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = np.random.randint(0, 2 ** 8, [period_int, 64, 128])
    current_period = period_int - np.random.rand(1)

    # determine resampling
    resample_factor = np.random.randint(3, 5)
    new_period = int(period_int * resample_factor)

    # resample with random factor
    resampled_sequence = hlp.resample_sequence(sequence, current_period, new_period)

    assert len(resampled_sequence) == new_period


def test_resample_sequence_2():
    # create a rectangular 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = np.random.randint(0, 2 ** 8, [period_int, 64, 128])

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(
        sequence, len(sequence), 2 * len(sequence)
    )

    assert np.all(resampled_sequence[::2, ...] == sequence)


def test_resample_sequence_1():
    # create a rectangular 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = np.random.randint(0, 2 ** 8, [period_int, 64, 128])

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(sequence, len(sequence), len(sequence))

    assert np.all(resampled_sequence == sequence)


def test_resample_sequence_known():
    # 'string' with triangular intensity pattern
    string = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    # use integer period
    period = 8

    # convert to uint8 'image sequence' (1x1 frame)
    sequence = np.asarray(string, "uint8").reshape([len(string), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence = np.repeat(np.repeat(sequence, 10, 1), 5, 2)

    # resample to 80
    resampled_sequence = hlp.resample_sequence(sequence, period, 80)

    right = []
    right.append(np.sum(resampled_sequence[:, 1, 1] == 4) == 1)
    right.append(np.sum(resampled_sequence[:, 1, 1] == 1) == 2 * 10)
    right.append(np.sum(resampled_sequence[:, 1, 1] == 2) == 2 * 10)
    right.append(np.sum(resampled_sequence[:, 1, 1] == 3) == 2 * 10)
    right.append(np.sum(resampled_sequence[:, 1, 1] == 0) == 80 - 3 * (2 * 10) - 1)

    assert np.all(right)


def test_cross_correlation_self():
    # create a rectangular 'image' sequence
    image = np.random.randint(0, 2 ** 8, [32, 64])
    period = 3
    sequence = np.tile(image[np.newaxis, ...], [period, 1, 1])

    # flatten each frame for 2D cross correlation
    sequence_flat = sequence.reshape([period, -1])

    # determine score with self
    scores = cc.cross_correlation(sequence_flat, sequence_flat)

    assert np.all(scores == 0)


def test_cross_correlation_notself():
    # create a rectangular 'image' sequence
    image = np.random.randint(0, 2 ** 8, [32, 64])
    period = 3
    sequence = np.tile(image[np.newaxis, ...], [period, 1, 1])

    # flatten each frame for 2D cross correlation
    sequence_flat = sequence.reshape([period, -1])

    # determine score with anti-self
    scores = cc.cross_correlation(sequence_flat, sequence_flat[::-1, ...])

    assert np.all(scores >= 0)


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
    # 'string' with sawtooth intensity pattern
    string1 = [0, 1, 2, 3, 4, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = 0 % period1
    string2 = np.roll(string1, roll)

    # convert to uint8 'image sequence' (1x1 frame)
    sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint8_equal_end():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 1, 2, 3, 4, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = len(string1) % period1
    string2 = np.roll(string1, roll)

    # convert to uint8 'image sequence' (1x1 frame)
    sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint8_equal_rand():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 1, 2, 3, 4, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = np.random.randint(1, len(string1) - 1) % period1
    string2 = np.roll(string1, roll)

    # convert to uint8 'image sequence' (1x1 frame)
    sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period1 = len(string1)
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint16_equal_start():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 1, 2, 3, 4, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = 0 % period1
    string2 = np.roll(string1, roll)

    # convert to uint8 'image sequence' (1x1 frame)
    sequence1 = np.asarray(string1, "uint16").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint16").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint16_equal_end():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 1, 2, 3, 4, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = len(string1) % period1
    string2 = np.roll(string1, roll)

    # convert to uint8 'image sequence' (1x1 frame)
    sequence1 = np.asarray(string1, "uint16").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint16").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint16_equal_rand():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 1, 2, 3, 4, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = np.random.randint(1, len(string1) - 1) % period1
    string2 = np.roll(string1, roll)

    # convert to uint8 'image sequence' (1x1 frame)
    sequence1 = np.asarray(string1, "uint16").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint16").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # small catch for floating point error
    assert roll_factor == roll or np.abs(roll_factor - roll) < 1e-6


def test_rolling_cross_correlation_uint8_notequal_start():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 3, 15, 63, 255, 63, 15, 3, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = 0 % period1
    string2 = np.roll(string1[:-1], roll)

    # convert to uint8 'image sequence' (1x1 frame)
    # type shouldn't matter for this
    sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint8_notequal_end():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 3, 15, 63, 255, 63, 15, 3, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = period1 % period1
    string2 = np.roll(string1[:-1], roll)

    # convert to uint8 'image sequence' (1x1 frame)
    # type shouldn't matter for this
    sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 (known result)
    assert roll_factor - 0.4469 < 1e-3


def test_rolling_cross_correlation_uint8_notequal_rand():
    # 'string' with sawtooth intensity pattern
    string1 = [0, 3, 15, 63, 255, 63, 15, 3, 0]
    # use integer periods
    period1 = len(string1)
    # shifted string (never rolls by zero or period)
    roll = np.random.randint(1, len(string1) - 1) % period1
    string2 = np.roll(string1[:-1], roll)

    # convert to uint8 'image sequence' (1x1 frame)
    # type shouldn't matter for this
    sequence1 = np.asarray(string1, "uint8").reshape([len(string1), 1, 1])
    sequence2 = np.asarray(string2, "uint8").reshape([len(string2), 1, 1])
    # convert to rectangular array (10x5 frame)
    sequence1 = np.repeat(np.repeat(sequence1, 10, 1), 5, 2)
    sequence2 = np.repeat(np.repeat(sequence2, 10, 1), 5, 2)
    # use integer periods
    period2 = len(string2)

    (_, _, roll_factor, _) = cc.rolling_cross_correlation(
        sequence1, sequence2, period1, period2
    )

    roll_factor = roll_factor % period2

    # catch a roll of ~0.4469 from intended (known result)
    assert roll_factor - roll - 0.4469 < 1e-3


