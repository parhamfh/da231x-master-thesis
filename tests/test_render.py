import numpy as np
import pytest

from sonata.render import _get_boundaries, beats_to_intervals, merge_intervals


@pytest.mark.xfail(reason="edge condition", strict=True)
def test_beats_to_intervals_bug():
    # For illustrative purposes:
    # If length of beat_samples is incorrect, off-by-1,
    # then last interval will have bad values, will pick first from `beat_sample`
    waveform = np.random.random(2077)
    beat_seq = [0, 1, 2, 6, 7, 9, 10]
    beat_samples = [
        111,
        222,
        333,
        444,
        # 555,  # This worked but gave bad intervals back
        666,
        777,
        888,
        999,
        1010,
        1111,
    ]
    intervals = beats_to_intervals(beat_seq, beat_samples, waveform)
    # should actually raise here that final beat in beat_seq i out of bounds for orig beat_samples.
    assert intervals[-1][1] != 111


def test_beats_to_intervals():
    waveform = np.random.random(2077)
    beat_seq = [0, 1, 2, 6, 7, 9, 10]
    beat_samples = [
        111,
        222,
        333,
        444,
        555,
        666,
        777,
        888,
        999,
        1010,
        1111,
    ]

    intervals = beats_to_intervals(beat_seq, beat_samples, waveform)
    assert len(intervals) == 3
    assert (intervals == [[0, 444], [777, 999], [1010, 2076]]).all()


def test_merge_intervals():
    # def
    iv = np.array(
        [
            [0, 111],
            [111, 222],
            [222, 333],
            [333, 444],
            [777, 888],
            [888, 999],
            [1010, 1111],
            [1111, 2076],
        ]
    )

    merge_iv = merge_intervals(iv[1:], iv[0])
    print(merge_iv)
    assert merge_iv == [[0, 444], [777, 999], [1010, 2076]]


def test_merge_interval_ensure_only_lists():
    intervals = np.array(
        [
            [0, 7680],
            [7680, 31680],
            [31680, 53760],
            [53760, 76800],
            [76800, 101760],
            [101760, 124800],
            [124800, 148800],
            [148800, 172800],
            [172800, 196799],
            [196799, 220799],
            [5067840, 5094720],
            [5094720, 5122560],
            [5122560, 5151360],
            [5151360, 5179200],
            [5179200, 5205120],
            [5205120, 5232960],
            [5232960, 5256960],
            [5256960, 5281920],
            [5281920, 5306880],
            [5306880, 5332800],
            [5332800, 5357760],
            [5357760, 5380800],
            [5380800, 5404800],
            [5404800, 5428800],
            [5428800, 5452800],
            [5452800, 5476800],
            [5476800, 5501760],
            [5501760, 5524800],
            [5524800, 5549760],
            [5645760, 5668800],
            [5764800, 5788800],
            [5788800, 5982774],
        ]
    )

    merged_iv = merge_intervals(intervals[1:], intervals[0])
    assert all([isinstance(iv, list) for iv in merged_iv])


def test_get_boundaries():
    intervals = [
        [0, 1],
        [2, 3],
        [77, 900],
        [900, 1008],  # Just a test case, real intervals are merged
        [2000, 3000],
    ]
    boundaries = _get_boundaries(intervals)

    assert boundaries == [
        (1, 2),
        (3, 77),
        (900, 900),
        (1008, 2000),
    ]
