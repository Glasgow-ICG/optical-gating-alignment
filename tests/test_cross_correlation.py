import optical_gating_alignment.cross_correlation as cc
import numpy as np


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


def test_rolling_cross_correlation_uint8_equal():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)
    period2 = period1

    accurate = []
    for roll in np.arange(period1):
        sequence2 = np.roll(sequence1, roll, axis=0)

        (_, _, roll_factor, _) = cc.rolling_cross_correlation(
            sequence1, sequence2, period1, period2
        )
        roll_factor = roll_factor % period2

        # small catch for floating point error
        accurate.append(np.abs(roll_factor - roll) < 1e6)

    assert np.all(accurate)


def test_rolling_cross_correlation_uint16_equal():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)
    period2 = period1

    accurate = []
    for roll in np.arange(period1):
        sequence2 = np.roll(sequence1, roll, axis=0)

        (_, _, roll_factor, _) = cc.rolling_cross_correlation(
            sequence1, sequence2, period1, period2
        )
        roll_factor = roll_factor % period2

        # small catch for floating point error
        accurate.append(np.abs(roll_factor - roll) < 1e6)

    assert np.all(accurate)


def test_rolling_cross_correlation_uint8_notequal():
    # create a rectangular uint8 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period1 = len(sequence1)

    accurate = []
    for roll in np.arange(period1):
        sequence2 = np.roll(sequence1, roll, axis=0)
        period2 = len(sequence2)

        (_, _, roll_factor, _) = cc.rolling_cross_correlation(
            sequence1, sequence2, period1, period2
        )
        roll_factor = roll_factor % period2

        # catch a roll of ~0.4469 (known result)
        accurate.append(np.abs(roll_factor - roll) < 0.45)

    assert np.all(accurate)


def test_rolling_cross_correlation_uint16_notequal():
    # create a rectangular uint16 'image' sequence with known values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period1 = len(sequence1)

    accurate = []
    for roll in np.arange(period1):
        sequence2 = np.roll(sequence1, roll, axis=0)
        period2 = len(sequence2)

        (_, _, roll_factor, _) = cc.rolling_cross_correlation(
            sequence1, sequence2, period1, period2
        )
        roll_factor = roll_factor % period2

        # catch a roll of ~0.4469 (known result)
        accurate.append(np.abs(roll_factor - roll) < 0.45)

    assert np.all(accurate)
