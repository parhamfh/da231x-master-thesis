from pathlib import Path

import numpy as np

from sonata.segmentation import _get_jams_path, segment_boundaries_to_beats


def test_get_jams_path():
    path = Path("my/audio/metal-tornado.wav")

    jams_path = _get_jams_path(path)

    assert jams_path == "my/references/metal-tornado.jams"


def test_segment_boundaries_to_beats():
    # beat_times = [0,2
    # segment_boundaries example, bossalova
    # array([  0.        ,   0.16      ,   9.96      ,  63.9       ,
    # 88.44      , 106.72      , 114.1       , 124.59827664,
    # 124.64117914])

    beat_times = np.array(
        [
            0.4,
            0.5,
            1,
            1.2,
            1.3,
            1.5,
            1.7,
            1.9,
            2,
            2.3,
            2.5,
            3.0,
            3.5,
            3.7,
            4.0,
            5,
            5.5,
            6,
        ]
    )
    segment_boundaries = np.array(
        [0, 0.4, 0.5, 1.5, 2.5, 4.0, 7.0, 9.0]
    )  # Last two will be ignored

    seg_beats = segment_boundaries_to_beats(segment_boundaries, beat_times)
    assert len(seg_beats) == 5
    assert seg_beats == [(0, 1), (1, 5), (5, 10), (10, 14), (14, 17)]
