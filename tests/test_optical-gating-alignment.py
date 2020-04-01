import optical_gating_alignment.optical_gating_alignment as oga
import test_helper as hlp
import numpy as np
from loguru import logger

logger.enable("optical_gating_alignment")


def test_process_sequence_cc_nodrift_uint8():
    sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    this_period = len(sequence)  # use integer
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        this_sequence = np.roll(sequence, roll, axis=0)
        # print(this_sequence[:, 0, 0])
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                _,
                this_target,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cc",
                method="fft",
                max_offset=3,
                resampled_period=80,
                ref_seq_id=0,
                ref_seq_phase=0,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                _,
                this_target,
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
                ref_seq_id=0,
                ref_seq_phase=0,
            )
        print(roll, 0.0, this_target)

        # collect results
        accurate.append(np.isclose(0, this_target))

    assert np.all(accurate)


def test_process_sequence_cc_nodrift_uint16():
    sequence = hlp.toy_sequence(
        seq_type="image", knowledge_type="known", dtype="uint16"
    )
    this_period = len(sequence)  # use integer
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        this_sequence = np.roll(sequence, roll, axis=0)
        # print(this_sequence[:, 0, 0])
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                _,
                this_target,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                algorithm="cc",
                method="fft",
                max_offset=3,
                resampled_period=80,
                ref_seq_id=0,
                ref_seq_phase=0,
            )
        else:
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                _,
                this_target,
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
                ref_seq_id=0,
                ref_seq_phase=0,
            )
        print(roll, 0.0, this_target)

        # collect results
        accurate.append(np.isclose(0, this_target))

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
                global_solution,
                roll_factor,
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
                global_solution,
                roll_factor,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                global_solution=global_solution,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        print(roll, 0.0, roll_factor)

        # collect results
        accurate.append(np.isclose(0.0, roll_factor))

    assert np.all(accurate)


# # FIXME JPS doesn't work with 16 bit images
# def test_process_sequence_cnw_nodrift_nointerp_uint16_sameintperiod():
#     sequence = hlp.toy_sequence(
#         seq_type="image", knowledge_type="known", dtype="uint16"
#     )
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         this_period = len(this_sequence)  # use integer
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 global_solution,
#                 roll_factor,
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
#                 global_solution,
#                 roll_factor,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 global_solution=global_solution,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         print(roll, 0.0, roll_factor)

#         # collect results
#         accurate.append(np.isclose(0.0, roll_factor))

#     assert np.all(accurate)


# TODO get interp working
# def test_process_sequence_cnw_nodrift_interp2_uint8_sameintperiod():
#     sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         this_period = len(this_sequence)  # use integer
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 global_solution,
#                 roll_factor,
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
#                 global_solution,
#                 roll_factor,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 global_solution=global_solution,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=2.0,
#             )
#         print(roll, 0.0, roll_factor)

#         # collect results
#         accurate.append(np.isclose(0.0, roll_factor))

#     assert np.all(accurate)


# def test_process_sequence_cnw_nodrift_interp2_uint16_sameintperiod():
#     sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         this_period = len(this_sequence)  # use integer
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 global_solution,
#                 roll_factor,
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
#                 global_solution,
#                 roll_factor,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 global_solution=global_solution,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=2.0,
#             )
#         print(roll, 0.0, roll_factor)

#         # collect results
#         accurate.append(np.isclose(0.0, roll_factor))

#     assert np.all(accurate)


def test_process_sequence_cnw_nodrift_nointerp_uint8_sameperiod():
    sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint8")
    this_drift = None  # [1, 1]  # TODO vary?

    accurate = []
    for roll in np.arange(10):
        this_sequence = np.roll(sequence, roll, axis=0)
        this_period = len(this_sequence) - 0.1
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                global_solution,
                roll_factor,
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
                global_solution,
                roll_factor,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                global_solution=global_solution,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        print(roll, 0.0, roll_factor)

        # collect results
        accurate.append(np.isclose(0.0, roll_factor))

    assert np.all(accurate)


# # FIXME JPS doesn't work with 16 bit images
# def test_process_sequence_cnw_nodrift_nointerp_uint8_sameperiod():
#     sequence = hlp.toy_sequence(seq_type="image", knowledge_type="known", dtype="uint16")
#     this_drift = None  # [1, 1]  # TODO vary?

#     accurate = []
#     for roll in np.arange(10):
#         this_sequence = np.roll(sequence, roll, axis=0)
#         this_period = len(this_sequence) - 0.1
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 global_solution,
#                 roll_factor,
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
#                 global_solution,
#                 roll_factor,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 global_solution=global_solution,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         print(roll, 0.0, roll_factor)

#         # collect results
#         accurate.append(np.isclose(0.0, roll_factor))

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
        this_period = len(this_sequence)
        if roll == 0:
            # test function can handle no history
            (
                sequence_history,
                period_history,
                drift_history,
                shift_history,
                global_solution,
                roll_factor,
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
                global_solution,
                roll_factor,
            ) = oga.process_sequence(
                this_sequence,
                this_period,
                this_drift,
                sequence_history=sequence_history,
                period_history=period_history,
                drift_history=drift_history,
                shift_history=shift_history,
                global_solution=global_solution,
                algorithm="cnw",
                ref_seq_id=0,
                ref_seq_phase=0,
                interpolation_factor=None,
            )
        print(roll, 0.0, roll_factor)

        # collect results
        accurate.append(np.isclose(0.0, roll_factor))

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
#         this_period = len(this_sequence)
#         if roll == 0:
#             # test function can handle no history
#             (
#                 sequence_history,
#                 period_history,
#                 drift_history,
#                 shift_history,
#                 global_solution,
#                 roll_factor,
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
#                 global_solution,
#                 roll_factor,
#             ) = oga.process_sequence(
#                 this_sequence,
#                 this_period,
#                 this_drift,
#                 sequence_history=sequence_history,
#                 period_history=period_history,
#                 drift_history=drift_history,
#                 shift_history=shift_history,
#                 global_solution=global_solution,
#                 algorithm="cnw",
#                 ref_seq_id=0,
#                 ref_seq_phase=0,
#                 interpolation_factor=None,
#             )
#         print(roll, 0.0, roll_factor)

#         # collect results
#         accurate.append(np.isclose(0.0, roll_factor))

#     assert np.all(accurate)
