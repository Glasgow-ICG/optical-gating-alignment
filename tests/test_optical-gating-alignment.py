import optical_gating_alignment.optical_gating_alignment as oga
import test_helper as hlp
import numpy as np


def test_process_sequence_cc_nodrift_uint8():
    sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    this_period = len(sequence)  # use integer
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        print(roll)
        this_sequence = np.roll(sequence, roll, axis=0)
        print(this_sequence[:, 0, 0])
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cc",
                method="fft",
                max_offset=3,
                resampled_period=80,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                algorithm="cc",
                method="fft",
                max_offset=3,
                resampled_period=80,
            )
        print(roll, roll_factor)
        # catch wrapping
        if roll - roll_factor > this_period / 2:
            roll_factor = roll_factor + this_period
        # catch small changes due to rolling point, interp and so on
        accurate.append(np.abs(roll - roll_factor) < 0.1)

    assert np.all(accurate)


def test_process_sequence_cc_nodrift_uint16():
    sequence = hlp.toy_sequence(
        seq_type="image", knowledge_type="known", dtype="uint16"
    )
    this_period = len(sequence)  # use integer
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        print(roll)
        this_sequence = np.roll(sequence, roll, axis=0)
        print(this_sequence[:, 0, 0])
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cc",
                method="fft",
                max_offset=3,
                resampled_period=80,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                algorithm="cc",
                method="fft",
                max_offset=3,
                resampled_period=80,
            )
        print(roll, roll_factor)
        # catch wrapping
        if roll - roll_factor > this_period / 2:
            roll_factor = roll_factor + this_period
        # catch small changes due to rolling point, interp and so on
        accurate.append(np.abs(roll - roll_factor) < 0.1)

    assert np.all(accurate)


def test_process_sequence_cnw_nodrift_nointerp_uint8_sameintperiod():
    sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        this_sequence = np.roll(sequence, roll, axis=0)
        this_period = len(this_sequence)  # use integer
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        print(roll, roll_factor)
        # catch wrapping
        if roll - roll_factor > this_period / 2:
            roll_factor = roll_factor + this_period
        # catch small changes due to rolling point, interp and so on
        accurate.append(np.abs(roll - roll_factor) < 0.1)

    assert np.all(accurate)


# # FIXME JPS doesn't work with 16 bit images
# def test_process_sequence_cnw_nodrift_nointerp_uint16_sameintperiod():
#     sequence = hlp.toy_sequence(
#         seq_type="image", knowledge_type="known", dtype="uint16"
#     )
#     this_period = len(sequence)  # use integer
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         else:
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         print(roll, roll_factor)
# # catch wrapping
# if roll - roll_factor > this_period/2:
#     roll_factor = roll_factor + this_period
#         # catch small changes due to rolling point, interp and so on
#         accurate.append(np.abs(roll - roll_factor) < 0.1)

#     assert np.all(accurate)


# TODO get interp working
# def test_process_sequence_cnw_nodrift_interp2_uint8_sameintperiod():
#     sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
#     this_period = len(sequence)  # use integer
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=2.0,
#             )
#         else:
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=2.0,
#             )
#         print(roll, roll_factor)
# # catch wrapping
# if roll - roll_factor > this_period/2:
#     roll_factor = roll_factor + this_period
#         # catch small changes due to rolling point, interp and so on
#         accurate.append(np.abs(roll - roll_factor) < 0.1)

#     assert np.all(accurate)


# def test_process_sequence_cnw_nodrift_interp2_uint16_sameintperiod():
#     sequence = hlp.toy_sequence(
#         seq_type="image", knowledge_type="known", dtype="uint16"
#     )
#     this_period = len(sequence)  # use integer
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=2.0,
#             )
#         else:
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=2.0,
#             )
#         print(roll, roll_factor)
# # catch wrapping
# if roll - roll_factor > this_period/2:
#     roll_factor = roll_factor + this_period
#         # catch small changes due to rolling point, interp and so on
#         accurate.append(np.abs(roll - roll_factor) < 0.1)

#     assert np.all(accurate)


def test_process_sequence_cnw_nodrift_nointerp_uint8_sameperiod():
    sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        this_sequence = np.roll(sequence, roll, axis=0)
        this_period = len(this_sequence) - 0.1  # use non integer
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        print(roll, roll_factor)
        # catch wrapping
        if roll - roll_factor > this_period / 2:
            roll_factor = roll_factor + this_period
        # catch small changes due to rolling point, interp and so on
        # catch bigger catches due to the uncertainty created by the non integer period
        accurate.append(np.abs(roll - roll_factor) < 0.5)

    assert np.all(accurate)


# # FIXME JPS doesn't work with 16 bit images
# def test_process_sequence_cnw_nodrift_nointerp_uint16_sameperiod():
#     sequence = hlp.toy_sequence(
#         seq_type="image", knowledge_type="known", dtype="uint16"
#     )
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#     this_period = len(this_sequence) - 0.1  # use non integer
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         else:
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         print(roll, roll_factor)
# # catch wrapping
# if roll - roll_factor > this_period/2:
#     roll_factor = roll_factor + this_period
#         # catch small changes due to rolling point, interp and so on
# # catch bigger catches due to the uncertainty created by the non integer period
# accurate.append(np.abs(roll - roll_factor) < 0.5)

#     assert np.all(accurate)


def test_process_sequence_cnw_nodrift_nointerp_uint8_diffintperiod():
    sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        this_sequence = np.roll(sequence, roll, axis=0)
        if roll == 2:
            # make one period a different length
            # this is similar to one arrhythmic sequence
            this_sequence = this_sequence[:-1]
        this_period = len(this_sequence)  # use integer
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                roll_factor,
                _,
                _,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        print(roll, roll_factor)
        # catch wrapping
        if roll - roll_factor > this_period / 2:
            roll_factor = roll_factor + this_period
        # catch small changes due to rolling point, interp and so on
        # catch bigger catches due to the uncertainty created by the differing period
        if roll != 2:
            # ignore the arrhythmic sequence as the sync will be off for that
            accurate.append(np.abs(roll - roll_factor) < 0.5)

    assert np.all(accurate)


# # FIXME JPS can't do 16-bit
# def test_process_sequence_cnw_nodrift_nointerp_uint16_diffintperiod():
#     sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         if roll == 2:
#             # make one period a different length
#             # this is similar to one arrhythmic sequence
#             this_sequence = this_sequence[:-1]
#         this_period = len(this_sequence)  # use integer
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         else:
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 roll_factor,
#                 _,
#                 _,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         print(roll, roll_factor)
#         # catch wrapping
#         if roll - roll_factor > this_period / 2:
#             roll_factor = roll_factor + this_period
#         # catch small changes due to rolling point, interp and so on
#         # catch bigger catches due to the uncertainty created by the differing period
#         if roll != 2:
#             # ignore the arrhythmic sequence as the sync will be off for that
#             accurate.append(np.abs(roll - roll_factor) < 0.5)

#     assert np.all(accurate)
