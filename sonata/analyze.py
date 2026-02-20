"""Audio analysis."""

import logging
from typing import Any, Literal

import librosa as lr
import numpy as np
import numpy.typing as npt
from BeatNet.BeatNet import BeatNet

logger = logging.getLogger(__name__)
DEBUG = True


def load_song(filename, sr):
    logger.info(f"Loading song. filename={filename}, sr={sr}")
    waveform, sr = lr.load(filename, sr=sr)

    return waveform, sr


def calculate_mfcc(waveform, sr):
    logger.info("Calculating MFCC features")
    mfcc = lr.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
    return mfcc


def calculate_cqt(waveform, sr):
    """Calculates the log power Constant-Q transform of the audio signal."""
    logger.info("Calculating CQT features")

    # Default values hop 512, fmin C1, 84 bins
    cqt = lr.cqt(waveform, sr=sr)
    power_cqt = np.abs(cqt) ** 2
    log_power_cqt = lr.power_to_db(power_cqt, ref=np.max)

    return log_power_cqt


def calculate_features(waveform, sr) -> dict[str, Any]:  # Correct type
    logger.info("Calculate features of waveform")
    mfcc = calculate_mfcc(waveform, sr)
    cqt = calculate_cqt(waveform, sr)

    logger.info(f"Calculated features. features={['mfcc', 'cqt']}")
    return {
        "mfcc": mfcc,
        "cqt": cqt,
    }


def analyze_beats(filename):
    logger.info(f"Analyze song beats with BeatNet. filename={filename}")
    estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False)
    beat_analysis = estimator.process(filename)
    beat_times, beat_numbers = np.split(beat_analysis, 2, axis=1)
    return beat_times.squeeze(), beat_numbers.squeeze()


def sync_features(features, beat_times, sr):
    logger.info("Sync features to beats")
    # (lr.time_to_frames(beats, sr=48000, hop_length=512) ==
    #  lr.time_to_frames(beats, sr=48000)).all()
    # Same hop_length for all features
    beat_frames = lr.time_to_frames(beat_times, sr=sr, hop_length=512)

    synced_features = {}
    for name, feature in features.items():
        logger.info(f"Syncing feature. feature={name}")
        fixed_frames = lr.util.fix_frames(beat_frames, x_min=None, x_max=feature.shape[1])
        synced_feature = lr.util.sync(feature, fixed_frames, pad=False)

        # Mega manic checking
        assert len(beat_times) == synced_feature.shape[1]
        assert (
            synced_feature[:, 0] == feature[:, fixed_frames[0] : fixed_frames[1]].mean(axis=1)
        ).all()
        assert (
            synced_feature[:, -1] == feature[:, fixed_frames[-2] : fixed_frames[-1]].mean(axis=1)
        ).all()

        synced_features[name] = synced_feature

    return synced_features


def compute_ssm(features, metric="gaussian"):
    logger.info(f"Compute self-similarity matrix using metric: {metric}")

    fs = ["mfcc", "cqt"]
    Rs = {}
    for fname in fs:
        logger.info(f"Computing SSM for feature. feature={fname}")
        f = features[fname]

        # dim_feat x num_beats
        if metric == "cosine":
            dot = np.dot(f.T, f)
            norm = np.linalg.norm(f, axis=0)
            ssm = dot / (norm[np.newaxis, :] * norm[:, np.newaxis])
        elif metric == "gaussian":
            # - sym=True - "to only link mutual nearest-neighbors"
            # Since we include all neighbours this does nothing, otherwise one of the pair might be
            # outside the other's knn and have cost 0
            # which the `minimum` call will replicate to the other cell.
            # - k is used for bandwidth even with full=True
            ssm = lr.segment.recurrence_matrix(
                f,
                k=f.shape[1],
                full=True,
                self=True,
                metric="cosine",
                mode="affinity",
                sym=True,
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if DEBUG:
            # assert all([np.isclose(ssm[i, i], 1) for i in range(len(ssm))])
            # Not true for lr.recurrence_matrix
            assert (ssm <= 1.000001).all()
            assert (ssm >= -1).all()

        Rs[fname] = ssm

    return Rs


def analyze_song(
    filename: str,
    song_name: str,
    sr: int,
    weights: list[float] | None = None,
    similarity: Literal["cosine", "gaussian"] = "gaussian",
):
    logger.info(f"Analyze song. filename={filename}, sr={sr}")
    waveform, sr = load_song(filename, sr)

    beat_times, beat_numbers = analyze_beats(filename)

    features = calculate_features(waveform, sr)
    synced_features = sync_features(features, beat_times, sr)

    Rs = compute_ssm(synced_features, metric=similarity)

    if weights is None:
        logger.info("Taking mean of feature matrices")
        R = np.mean([Rs["mfcc"], Rs["cqt"]], axis=0)
    else:
        logger.info(f"Combining feature matrices with weights. weights={weights}")
        R = np.average([Rs["mfcc"], Rs["cqt"]], weights=weights, axis=0)

    if DEBUG:
        keys = ["mfcc", "cqt"]
        logger.info(
            f"max and min values. keys={keys},"
            f" max={np.max(np.array([Rs[k] for k in keys]), axis=(1, 2))},"
            f" min={np.min(np.array([Rs[k] for k in keys]), axis=(1, 2))}"
        )
        logger.info(f"max and min in final R. max={np.max(R)}, min={np.min(R)}")
    return waveform, beat_times, beat_numbers, R, Rs


def average_bpm(beat_times: npt.NDArray[np.float32]) -> np.float32:
    logger.info("Calculate average BPM of song")
    return 60 / np.diff(beat_times).mean()


def seconds_to_beats(duration_seconds: float, bpm: np.float32) -> int:
    beat_duration_s = bpm / 60
    target_num_beats = int(np.round(duration_seconds * beat_duration_s))
    logger.info(
        f"Converted seconds to number of beats. duration_seconds={duration_seconds},"
        f" bpm={bpm}, target_num_beats={target_num_beats}"
    )
    # Round because graph is over discrete beats
    return target_num_beats
