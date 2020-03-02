import optical_gating_alignment as oga
import optical_gating_alignment.helper as hlp
from optical_gating_alignment.helper import drift_correction
import numpy as np


def test_version():
    assert oga.__version__ == "2.0.0"


def test_drift_correction_uint8():
    # create a rectangular uint8 'image sequence'
    image1 = np.random.randint(0, 2 ** 8, [64, 128]).astype("uint8")
    sequence1 = image1[np.newaxis, ...]  # a sequence of length 1

    # create a rolled, i.e. drifted, version
    drift_x = np.random.randint(1, 5)
    drift_y = np.random.randint(1, 5)
    image2 = np.roll(image1, drift_x, axis=0)
    image2 = np.roll(image2, drift_y, axis=1)
    sequence2 = image2[np.newaxis, ...]

    # correct for drift
    corrected1, corrected2 = drift_correction(sequence1, sequence2, [drift_x, drift_y])

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
    corrected1, corrected2 = drift_correction(sequence1, sequence2, [drift_x, drift_y])

    assert np.all(corrected1[0] == corrected2[0])


def test_resample_sequence():
    # create a rectangular 'image' sequence
    image = np.random.randint(0, 2 ** 8, [64, 128])
    period_int = np.random.randint(5, 11)
    sequence = np.tile(image[np.newaxis, ...], [period_int, 1, 1])
    current_period = period_int - np.random.rand(1)

    # determine resampling
    resample_factor = np.random.randint(1, 3)
    new_period = int(current_period * resample_factor)

    # resample with random factor
    resampled_sequence = hlp.resample_sequence(sequence, current_period, new_period)

    assert len(resampled_sequence) == new_period

    # resample with factor of 1
    resampled_sequence = hlp.resample_sequence(sequence, len(sequence), len(sequence))

    assert np.all(resampled_sequence == sequence)
