import numpy as np
import pytest

from sonata.evaluate import (
    count_backjumps,
    section_proportion_error,
    track_coverage,
    weighted_section_recall,
)


def test_count_backjumps():
    # With corner cases, usually starts on 0 and on last actual beat
    b_seq = [10, 2, 3, 4, 5, 19, 10, 2, 4, 6, 20, 15, 12]

    assert count_backjumps(b_seq) == 5


def test_track_coverage():
    b_seq = list(range(0, 10))
    num_beats = 20

    assert track_coverage(b_seq, num_beats) == 0.5

    # With gaps
    b_seq = list(range(3, 16, 2))
    num_beats = 20
    assert track_coverage(b_seq, num_beats) == 7 / 20

    # Irrespective of order
    b_seq = [9, 6, 3, 2]
    num_beats = 10
    assert track_coverage(b_seq, num_beats) == 4 / 10


def test_weighted_section_recall_1():
    assert 1 == weighted_section_recall(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))


def test_weighted_section_recall_scaled():
    assert 0.5 == weighted_section_recall(np.array([0, 1, 2]), np.array([0, 1, 2, 3, 3, 3]))

    # Beat sequence needs to have the right colors
    beat_seq = np.array([0, 2, 4])
    beat_colors = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
    assert np.isclose(0.6, weighted_section_recall(beat_seq, beat_colors))


# flake8: noqa
# fmt: off
def test_weighted_section_recall_real_example():
    # Example values from pineapple.wav
    beat_sequence = np.array([  0,   1,  26,  27,  28,  29,  30,  31,  32,  33,  34, 315, 316,
       317, 318, 319, 320, 321, 322, 323, 324, 325, 334, 335, 336, 337,
       338, 339, 340, 341, 342], dtype=np.int32)

    beat_colors = np.array([
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32)

    # sequence_colors = array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=int32)
    # unique colors = {1, 2, 3}

    wsr = weighted_section_recall(beat_sequence, beat_colors)

    assert np.isclose(wsr, 0.95335)



ranger = lambda x: np.arange(x, dtype=np.int32)
painter = lambda x, y: np.concatenate([np.ones(x, np.int32), np.zeros(y, dtype=np.int32)])

@pytest.mark.parametrize("beat_seq,beat_colors,expected",[
    [ranger(6), painter(6,0), 0.0],
    [ranger(2), painter(1,9), 0.8],
    [ranger(2), painter(1,99), 0.98],
    [ranger(100), painter(100,100000), 1.998],
    [ranger(100), painter(99,99999), 1.978],
    [np.array([0,1]), np.array([0,1,2], dtype=np.int32), 0.667],  # not all colors present in sequence

])
def test_section_proportion_error(beat_seq, beat_colors, expected):
    sp_err = section_proportion_error(beat_seq, beat_colors)
    assert round(sp_err,3) == expected


def test_section_proportion_error_bossa_lova():
    # Excessive defensive testing with data from a remix of Bossa Lova
    beat_seq = np.array([  0,   1,   2, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
       181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
       194, 195, 196, 197, 198], dtype=np.int32)
    beat_colors = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3], dtype=np.int32)
    err = section_proportion_error(beat_seq, beat_colors)

    assert err == 1.1457286432160805
