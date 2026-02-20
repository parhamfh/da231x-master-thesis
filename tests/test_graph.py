from itertools import cycle, islice

import numpy as np
import pytest

from sonata.graph import _get_penalty_matrix, add_back_jump_penalty, stay_on_meter


@pytest.fixture
def data():
    pass


def test_stay_on_meter():
    # Setup
    # Bossa Lova example, starts on 3rd beat
    num_beats = 199
    beat_numbers = np.array(list(islice(cycle([3.0, 4.0, 1.0, 2.0]), num_beats)))
    btm = np.zeros((num_beats, num_beats))

    m_btm = stay_on_meter(btm, beat_numbers, full_measures=False)

    exp_0 = np.tile(np.array([100, 0, 100, 100]), num_beats // 3)
    expected = np.array(
        [
            exp_0[:num_beats],
            np.roll(exp_0, 1)[:num_beats],
            np.roll(exp_0, 2)[:num_beats],
            np.roll(exp_0, 3)[:num_beats],
        ]
    )
    assert (m_btm[:4] == expected).all()


def test_stay_on_meter_only_downbeat():
    # Setup
    # Bossa Lova example, starts on 3rd beat
    num_beats = 199
    beat_numbers = np.array(list(islice(cycle([3.0, 4.0, 1.0, 2.0]), num_beats)))
    btm = np.zeros((num_beats, num_beats))

    m_btm = stay_on_meter(btm, beat_numbers, full_measures=True)

    exp_0 = np.full(num_beats, 100)
    # ok transition
    exp_0[1] = 0
    exp_1 = np.tile(np.array([100, 100, 0, 100]), num_beats // 3)
    expected = np.array(
        [
            exp_0,
            exp_1[:num_beats],
            np.roll(exp_0, 2)[:num_beats],
            np.roll(exp_0, 3)[:num_beats],
        ]
    )
    assert (m_btm[:4] == expected).all()


def test_get_penalty_matrix():
    w = 3
    p_mx = _get_penalty_matrix(4, w=3)
    print(p_mx)
    assert (
        p_mx
        == np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [w / 3, 1, 0, 0],
                [w / 4, w / 3, 1, 0],
            ]
        )
    ).all()


def test_add_back_jump_penalty():
    btm = np.ones((10, 10))
    p_mx = _get_penalty_matrix(10)

    assert (btm + p_mx == add_back_jump_penalty(btm)).all()


@pytest.mark.skip
def test_get_btm(data):
    pass


@pytest.mark.skip
def test_create_graph(data):
    pass
