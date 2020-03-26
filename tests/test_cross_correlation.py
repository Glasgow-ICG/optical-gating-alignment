import optical_gating_alignment.cross_correlation as cc
import numpy as np
from loguru import logger

logger.enable("optical_gating_alignment")


def toy_sequence(length=0, seq_type="image", knowledge_type="random", dtype="uint8"):
    if knowledge_type == "random" and seq_type == "string":
        sequence = np.random.randint(0, 2 ** 8, length).astype(dtype)
    elif knowledge_type == "random" and seq_type == "image":
        sequence = np.random.randint(0, 2 ** 8, [length, 64, 128]).astype(dtype)
    elif knowledge_type == "known":
        # 'string' with uint8 triangular intensity pattern
        string = [0, 3, 15, 63, 255, 63, 15, 3, 0]
        if seq_type == "string":
            # convert to uint8
            sequence = np.asarray(string, dtype)
        elif seq_type == "image":
            # convert to uint8 'image sequence' (1x1 frame)
            sequence = np.asarray(string, dtype).reshape([len(string), 1, 1])
            # convert to rectangular array (64x128 frame)
            sequence = np.repeat(np.repeat(sequence, 64, 1), 128, 2).astype(dtype)
    elif seq_type == "alignment":
        sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return sequence


def test_cross_correlation_uint8_self():
    # create a rectangular uint8 'image' sequence where each pixel is random
    sequence = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    print(sequence[:, 0, 0])

    # determine score with self - SSD
    ssd_scores = cc.cross_correlation(sequence, sequence.copy(), method="ssd")
    print(ssd_scores)

    # determine score with self - SSD
    fft_scores = cc.cross_correlation(sequence, sequence.copy(), method="fft")
    print(fft_scores)

    # determine score with self - SSD
    jps_scores = cc.cross_correlation(sequence, sequence.copy(), method="jps")
    print(jps_scores)

    assert (
        np.all(ssd_scores == fft_scores)
        and np.all(ssd_scores == jps_scores)
        and np.all(ssd_scores >= 0)
    )


def test_cross_correlation_uint16_self():
    # create a rectangular uint16 'image' sequence where each pixdl is random
    sequence = toy_sequence(
        length=10, seq_type="image", knowledge_type="known", dtype="uint16"
    )
    print(sequence[:, 0, 0])

    # determine score with self - SSD
    ssd_scores = cc.cross_correlation(sequence, sequence.copy(), method="ssd")
    print(ssd_scores)

    # determine score with self - SSD
    fft_scores = cc.cross_correlation(sequence, sequence.copy(), method="fft")
    print(fft_scores)

    # determine score with self - SSD
    jps_scores = cc.cross_correlation(sequence, sequence.copy(), method="jps")
    print(jps_scores)

    assert (
        np.all(ssd_scores == fft_scores)
        and np.all(ssd_scores == jps_scores)
        and np.all(ssd_scores >= 0)
    )


def test_cross_correlation_uint8_notself():
    # create two rectangular uint8 'image' sequences where each pixel is random
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    sequence2 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    print(sequence1[:, 0, 0])
    print(sequence2[:, 0, 0])

    # determine score with self - SSD
    ssd_scores = cc.cross_correlation(sequence1, sequence2, method="ssd")
    print(ssd_scores)

    # determine score with self - SSD
    fft_scores = cc.cross_correlation(sequence1, sequence2, method="fft")
    print(fft_scores)

    # determine score with self - SSD
    jps_scores = cc.cross_correlation(sequence1, sequence2, method="jps")
    print(jps_scores)

    assert (
        np.all(ssd_scores == fft_scores)
        and np.all(ssd_scores == jps_scores)
        and np.all(ssd_scores >= 0)
    )


def test_cross_correlation_uint16_notself():
    # create two rectangular uint16 'image' sequences where each pixel is random
    sequence1 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    sequence2 = toy_sequence(
        length=10, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    print(sequence1[:, 0, 0])
    print(sequence2[:, 0, 0])

    # determine score with self - SSD
    ssd_scores = cc.cross_correlation(sequence1, sequence2, method="ssd")
    print(ssd_scores)

    # determine score with self - SSD
    fft_scores = cc.cross_correlation(sequence1, sequence2, method="fft")
    print(fft_scores)

    # determine score with self - SSD
    jps_scores = cc.cross_correlation(sequence1, sequence2, method="jps")
    print(jps_scores)

    assert (
        np.all(ssd_scores == fft_scores)
        and np.all(ssd_scores == jps_scores)
        and np.all(ssd_scores >= 0)
    )


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
    print(minimum_position, minimum_value)
    # the 1.0+ is because this function returns position from the start
    assert minimum_position == 1.0 + 0.0 and minimum_value == 0.1


def test_minimum_score_complex():
    # slightly more complex example
    scores = [3.5, 1.1, 2.0]
    minimum_position, minimum_value = cc.minimum_score(scores)
    print(minimum_position, minimum_value)
    # the round in the line below is because of floating point error
    assert minimum_position == 1.0 + 0.3125 and np.round(minimum_value, 2) == 0.35


def test_minimum_score_flipped():
    # slightly more complex example - flipped
    scores = [2.0, 1.1, 3.5]
    minimum_position, minimum_value = cc.minimum_score(scores)
    print(minimum_position, minimum_value)
    # the round in the line below is because of floating point error
    assert minimum_position == 1.0 - 0.3125 and np.round(minimum_value, 2) == 0.35


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
            sequence1, sequence2, period1, period2,
        )

        print(roll, roll_factor, period2)
        # small catch for floating point and other errors
        # FIXME this seems to be very high
        accurate.append(np.abs(roll_factor - roll) < 0.25)

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
            sequence1, sequence2, period1, period2,
        )
        roll_factor = roll_factor % period2

        print(roll, roll_factor)
        # small catch for floating point and other errors
        # FIXME this seems to be very high
        accurate.append(np.abs(roll_factor - roll) < 0.25)

    assert np.all(accurate)


def test_rolling_cross_correlation_uint8_notequal():
    # this test could be better - it doesn't fill one with confidence
    # create a rectangular uint8 'image' sequence with random values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="random", dtype="uint8")
    # use integer period
    period1 = len(sequence1)

    accurate = []
    for roll in np.arange(period1):
        sequence2 = np.roll(sequence1[:-2], roll, axis=0)
        period2 = len(sequence2)

        (_, _, roll_factor, _) = cc.rolling_cross_correlation(
            sequence1, sequence2, period1, period2,
        )
        roll_factor = roll_factor % period2

        print(roll, roll_factor)
        # within +/-1 of roll seems reasonable
        # FIXME this seems to be very high
        if np.abs(roll_factor - roll) > period1 / 2:
            print("account for wrapping", roll - period1, roll_factor)
            accurate.append(
                np.abs(roll_factor - (roll - period1)) < 2 + 0.1
            )  # bigger due to missing points
        else:
            accurate.append(np.abs(roll_factor - roll) < 0.1)
        print(accurate[-1])

    assert np.all(accurate)


def test_rolling_cross_correlation_uint16_notequal():
    # this test could be better - it doesn't fill one with confidence
    # create a rectangular uint16 'image' sequence with random values
    sequence1 = toy_sequence(seq_type="image", knowledge_type="random", dtype="uint16")
    # use integer period
    period1 = len(sequence1)

    accurate = []
    for roll in np.arange(period1):
        sequence2 = np.roll(sequence1[:-2], roll, axis=0)
        period2 = len(sequence2)

        (_, _, roll_factor, _) = cc.rolling_cross_correlation(
            sequence1, sequence2, period1, period2,
        )
        roll_factor = roll_factor % period2

        print(roll, roll_factor)
        # within +/-1 of roll seems reasonable
        # FIXME this seems to be very high
        if np.abs(roll_factor - roll) > period1 / 2:
            print("account for wrapping", roll - period1, roll_factor)
            accurate.append(
                np.abs(roll_factor - (roll - period1)) < 2 + 0.1
            )  # bigger due to missing points
        else:
            accurate.append(np.abs(roll_factor - roll) < 0.1)
        print(accurate[-1])

    assert np.all(accurate)
