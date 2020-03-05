import optical_gating_alignment.helper as hlp
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


def test_linear_interpolation_uint8():
    # create a rectangular uint8 'image' sequence
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")

    accurate = []
    for interp_pos in np.arange(0.5, len(sequence) - 0.5, 1):
        interpolated_value = hlp.linear_interpolation(
            sequence, interp_pos, period=None
        )[0, 0]
        # astype('int') is needed to prevent wrapping
        known_value = int(
            (
                sequence[int(interp_pos), 0, 0].astype("int")
                + sequence[int(interp_pos + 1), 0, 0].astype("int")
            )
            / 2
        )
        accurate.append(interpolated_value == known_value)

    assert np.all(accurate)


def test_interpolate_image_sequence_uint8_period():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use non-integer period
    current_period = period_int - np.random.rand(1)

    accurate = []
    for resample_factor in np.arange(5):
        # resample
        resampled_sequence = hlp.interpolate_image_sequence(
            sequence, current_period, resample_factor
        )

        accurate.append(
            len(resampled_sequence) == int(current_period * resample_factor) + 1
        )

    assert np.all(accurate)


def test_interpolate_image_sequence_uint8_2():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.interpolate_image_sequence(sequence, len(sequence), 2.0)

    assert np.all(resampled_sequence[::2, ...] == sequence)


def test_interpolate_image_sequence_uint8_1():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.interpolate_image_sequence(sequence, len(sequence), 1.0)

    assert np.all(resampled_sequence == sequence)


def test_interpolate_image_sequence_uint16_period():
    # create a rectangular uint16 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    # use non-integer period
    current_period = period_int - np.random.rand(1)

    accurate = []
    for resample_factor in np.arange(5):
        # resample
        resampled_sequence = hlp.interpolate_image_sequence(
            sequence, current_period, resample_factor
        )

        accurate.append(
            len(resampled_sequence) == int(resample_factor * current_period) + 1
        )

    assert np.all(accurate)


def test_interpolate_image_sequence_uint16_2():
    # create a rectangular uint16 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.interpolate_image_sequence(sequence, len(sequence), 2.0)

    assert np.all(resampled_sequence[::2, ...] == sequence)


def test_interpolate_image_sequence_uint16_1():
    # create a rectangular uint16 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint16"
    )
    # use integer period

    # resample with factor of 1
    resampled_sequence = hlp.interpolate_image_sequence(sequence, len(sequence), 1.0)

    assert np.all(resampled_sequence == sequence)


def test_interpolate_image_sequence_uint8_known():
    # create a rectangular uint8 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.interpolate_image_sequence(sequence, period, 80 / period)

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


def test_interpolate_image_sequence_uint16_known():
    # create a rectangular uint16 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.interpolate_image_sequence(sequence, period, 80 / period)

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
