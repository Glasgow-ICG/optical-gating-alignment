import optical_gating_alignment as oga
from optical_gating_alignment.helper import drift_correction
import numpy as np


def test_version():
    assert oga.__version__ == "2.0.0"


def test_drift_correction_uint8():
    # create a rectangular uint8 'image'
    image1 = np.random.random_integers(0, 255, [64, 128]).astype("uint8")
    sequence1 = image1[np.newaxis, ...]  # a sequence of length 1

    # create a rolled, i.e. drifted, version
    drift_x = np.random.random_integers(1, 5)
    drift_y = np.random.random_integers(1, 5)
    image2 = np.roll(image1, drift_x, axis=0)
    image2 = np.roll(image2, drift_y, axis=1)
    sequence2 = image2[np.newaxis, ...]

    # correct for drift
    corrected1, corrected2 = drift_correction(sequence1, sequence2, [drift_x, drift_y])

    assert np.all(corrected1[0] == corrected2[0])


def test_drift_correction_uint16():
    # create a rectangular uint16 'image'
    image1 = np.random.random_integers(0, 255, [64, 128]).astype("uint16")
    sequence1 = image1[np.newaxis, ...]  # a sequence of length 1

    # create a rolled, i.e. drifted, version
    drift_x = np.random.random_integers(1, 5)
    drift_y = np.random.random_integers(1, 5)
    image2 = np.roll(image1, drift_x, axis=0)
    image2 = np.roll(image2, drift_y, axis=1)
    sequence2 = image2[np.newaxis, ...]

    # correct for drift
    corrected1, corrected2 = drift_correction(sequence1, sequence2, [drift_x, drift_y])

    assert np.all(corrected1[0] == corrected2[0])
