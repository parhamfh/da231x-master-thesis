import logging

import librosa as lr
import numpy as np
import numpy.typing as npt
import soundfile as sf
from librosa.effects import remix

from sonata.types import Waveform

logger = logging.getLogger(__name__)


def merge_intervals(intervals, current_interval) -> list[list[int]]:
    """
    intervals           - numpy array of intervals
    current_interval    - any list type interval

    Returns: list of intervals

    Feed it like this (Haskell style):
    ```
    intervals =  [...]
    merge_intervals(intervals[1:], intervals[0])
    ```
    """
    # Could rap this function for simpler API
    if intervals.size == 0:
        return [current_interval]

    # Next interval in list of intervals
    interval = intervals[0]

    # are they connected?
    if current_interval[1] == interval[0]:
        current_interval = [current_interval[0], interval[1]]
        return merge_intervals(intervals[1:], current_interval)
    else:
        # In the case interval is an intact row from the intervals we need to convert type
        new_interval = [list(current_interval)]
        new_interval.extend(merge_intervals(intervals[1:], interval))
        return new_interval


def beats_to_intervals(
    beat_sequence: npt.NDArray[np.int32],
    beat_samples: npt.NDArray[np.int32],
    waveform: Waveform,
) -> npt.NDArray:
    """Convert sequence of beats to sample intervals.

    Method assumes beat_sequence starts with first beat of song and ends
    with last
    """
    # Handle last beat: assuming beat_sequence ends with last beat of original audio,
    beat_samples = np.concatenate(
        [beat_samples, [len(waveform) - 1]]
    )  # add last sample for last beat
    beat_end_frames = np.roll(beat_samples, -1)
    intervals = []
    for beat in beat_sequence:
        intervals.append([beat_samples[beat], beat_end_frames[beat]])

    # Handle first beat
    if beat_samples[0] != 0:
        # Manually add intro samples not counted into first beat by beat tracking
        logger.info(
            f"First beat sample not zero, adding preceding samples."
            f" beat_samples[0]={beat_samples[0]}"
        )
        intervals = np.insert(intervals, 0, [0, beat_samples[0]], axis=0)  # type: ignore
    else:
        # Manually convert to numpy.ndarray which remix requires
        intervals = np.array(intervals)  # type: ignore

    logger.info(f"Calculated frame intervals. intervals={intervals}")
    merged_intervals = merge_intervals(intervals[1:], intervals[0])
    logger.info(f"Merged frame intervals. intervals={merged_intervals}")
    return np.asarray(merged_intervals)


def remix_song(intervals, waveform: Waveform) -> Waveform:
    # No align_zeros since we crossfade later
    path_y = remix(waveform, intervals, align_zeros=False)

    return path_y


def _get_boundaries(intervals):
    bounds = []
    for i in range(len(intervals) - 1):
        bounds.append((intervals[i][1], intervals[i + 1][0]))
    return bounds


def apply_crossfade(remix_waveform, intervals, waveform, sr, crossfade):
    logger.info(f"Applying crossfade. crossfade={crossfade}")
    # Size of crossfade window - ensure it is symmetric
    cf_size = lr.time_to_samples(crossfade, sr=sr)
    cf_size = cf_size + 1 if cf_size % 2 != 0 else cf_size
    cf_window = cf_size // 2

    ramp_down = [1 - (n / (cf_size - 1)) for n in range(cf_size)]
    ramp_up = [n / (cf_size - 1) for n in range(cf_size)]

    boundaries = _get_boundaries(intervals)
    rmx_ix = 0
    for ix, (A, B) in enumerate(boundaries):
        logger.info(f"Cross-fading boundary {ix} between frames {A} and {B}")

        # Length current interval, indexing into remix_waveform
        ivl = intervals[ix]
        ivl_len = ivl[1] - ivl[0]
        # Boundary location
        rmx_ix += ivl_len

        # Note-to-self: `lr.remix` treats end of interval as exclusive

        # Consider segment frames/samples A and B at boundary: A | B
        remix_waveform[rmx_ix - cf_window : rmx_ix + cf_window] = (
            waveform[A - cf_window : A + cf_window] * ramp_down
            + waveform[B - cf_window : B + cf_window] * ramp_up
        )


def write_audio(waveform: Waveform, filename: str, output_dir) -> None:
    out_file = f"{output_dir}/{filename}.mp3"
    sf.write(out_file, waveform, 48000)
    logger.info(f"Rendering Generated song saved on disk. filename={out_file}")


def render_remix(
    intervals,
    waveform,
    sr,
    output_dir,
    filename,
    crossfade: float = 0.5,
    dur_n_tol=None,
) -> int:
    """
    crossfade   - duration of crossfade in sec
    """
    remix_waveform = remix_song(intervals, waveform)
    if crossfade > 0:
        apply_crossfade(remix_waveform, intervals, waveform, sr, crossfade)

    logger.info(f"Song remixed. duration={int(remix_waveform.shape[0] / sr)}")
    remix_duration = int(remix_waveform.shape[0] / sr)
    if dur_n_tol:
        target_dur, tol_sec = dur_n_tol
        logger.info(f"remix duration={remix_duration}, target_dur={target_dur}, tol_sec={tol_sec}")
        if not target_dur - tol_sec <= remix_duration <= target_dur + tol_sec:
            logger.warn("Remixed track exceeds duration limits")

    write_audio(remix_waveform, filename, output_dir)
    return remix_duration
