import optical_gating_alignment.helper as hlp
import numpy as np


# TODO: check float outputs from interp


def toy_sequence(length=0, seq_type="image", knowledge_type="random", dtype="uint8"):
    if seq_type == "image":
        # note period is length or less (10 or less for know; 5 or less for tiny)
        if knowledge_type == "random":
            sequence = np.random.randint(0, 2 ** 8, [length, 64, 128]).astype(dtype)
        elif knowledge_type == "known":
            # 'string' with uint8 triangular intensity pattern
            string = [1, 3, 15, 63, 255, 64, 16, 4, 2]
            # convert to uint8 'image sequence' (1x1 frame)
            sequence = np.asarray(string, "uint8").reshape([len(string), 1, 1])
            # convert to rectangular array (64x128 frame)
            sequence = np.repeat(np.repeat(sequence, 64, 1), 128, 2).astype(dtype)
        elif knowledge_type == "tiny":
            # 'string' with uint8 triangular intensity pattern
            string = [1, 63, 255, 64, 2]
            # convert to uint8 'image sequence' (1x1 frame)
            sequence = np.asarray(string, "uint8").reshape([len(string), 1, 1])
            # convert to rectangular array (64x128 frame)
            sequence = np.repeat(np.repeat(sequence, 64, 1), 128, 2).astype(dtype)
    elif seq_type == "alignment":
        # note: period is 9 or less
        sequence = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
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


def test_interpolate_image_sequence_uint8_period():
    # create a rectangular uint8 'image' sequence
    period_int = np.random.randint(5, 11)
    sequence = toy_sequence(
        length=period_int, seq_type="image", knowledge_type="random", dtype="uint8"
    )
    # use non-integer period
    current_period = period_int - np.random.rand(1)

    accurate = []
    for resample_factor in np.arange(1, 5):
        # resample
        resampled_sequence = hlp.interpolate_image_sequence(
            sequence, current_period, interpolation_factor=resample_factor
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
    for resample_factor in np.arange(1, 5):
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


def test_interpolate_image_sequence_uint8_known_uint8():
    # assumes the code was correct at the time this test was made
    # create a rectangular uint8 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.interpolate_image_sequence(
        sequence, period, 80 / period, dtype="uint8"
    )

    # this was very manual
    expected = [
        1,
        1,
        1,
        1,
        1,
        2,
        2,
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
        140,
        121,
        102,
        83,
        64,
        59,
        54,
        49,
        44,
        40,
        35,
        30,
        25,
        20,
        16,
        14,
        13,
        12,
        11,
        10,
        8,
        7,
        6,
        5,
        4,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
    ]

    print(sequence[:, 1, 1], resampled_sequence[:, 1, 1])
    assert (
        np.all(expected == resampled_sequence[:, 1, 1])
        and resampled_sequence.dtype == np.uint8
    )


def test_interpolate_image_sequence_uint16_known_uint16():
    # assumes the code was correct at the time this test was made
    # create a rectangular uint16 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.interpolate_image_sequence(
        sequence, period, 80 / period, dtype="uint16"
    )

    # this was very manual
    expected = [
        1,
        1,
        1,
        1,
        1,
        2,
        2,
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
        140,
        121,
        102,
        83,
        64,
        59,
        54,
        49,
        44,
        40,
        35,
        30,
        25,
        20,
        16,
        14,
        13,
        12,
        11,
        10,
        8,
        7,
        6,
        5,
        4,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
    ]

    print(sequence[:, 1, 1], resampled_sequence[:, 1, 1])
    assert (
        np.all(expected == resampled_sequence[:, 1, 1])
        and resampled_sequence.dtype == np.uint16
    )


def test_interpolate_image_sequence_uint8_known_float():
    # assumes the code was correct at the time this test was made
    # create a rectangular uint8 'image' sequence with known values
    sequence = toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    # use integer period
    period = 8

    # resample to 80
    resampled_sequence = hlp.interpolate_image_sequence(
        sequence, period, 80 / period, dtype="float"
    )

    # this was very manual
    expected = [
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        4.2,
        5.4,
        6.6,
        7.8,
        9.0,
        10.2,
        11.4,
        12.6,
        13.8,
        15.0,
        19.8,
        24.6,
        29.4,
        34.2,
        39.0,
        43.8,
        48.6,
        53.4,
        58.2,
        63.0,
        82.2,
        101.4,
        120.6,
        139.8,
        159.0,
        178.2,
        197.4,
        216.6,
        235.8,
        255.0,
        235.9,
        216.8,
        197.7,
        178.6,
        159.5,
        140.4,
        121.3,
        102.2,
        83.1,
        64.0,
        59.2,
        54.4,
        49.6,
        44.8,
        40.0,
        35.2,
        30.4,
        25.6,
        20.8,
        16.0,
        14.8,
        13.6,
        12.4,
        11.2,
        10.0,
        8.8,
        7.6,
        6.4,
        5.2,
        4.0,
        3.8,
        3.6,
        3.4,
        3.2,
        3.0,
        2.8,
        2.6,
        2.4,
        2.2,
    ]

    print(sequence[:, 1, 1], resampled_sequence[:, 1, 1])
    assert (
        np.all(np.abs(expected - resampled_sequence[:, 1, 1]) <= 1e-6)
        and resampled_sequence.dtype == np.float
    )
