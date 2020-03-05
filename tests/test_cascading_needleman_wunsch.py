import optical_gating_alignment.cascading_needleman_wunsch as cnw
import numpy as np


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
