import logging
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Literal

import librosa as lr

from sonata.analyze import analyze_song, average_bpm, seconds_to_beats
from sonata.evaluate import evaluate_remix
from sonata.graph import create_graph, shortest_path
from sonata.metadata import Timer, get_run_metadata, save_metadata
from sonata.plots import save_R_plot
from sonata.render import beats_to_intervals, render_remix
from sonata.segmentation import (
    _cache_seg,
    _check_seg_cache,
    save_beats_jams,
    segment,
    segment_boundaries_to_beats,
)

OUTPUT_DIR = "output"
SAMPLE_RATE = 48000
DEFAULT_TOL_SEC = 4
DEFAULT_CROSSFADE = 0.3
DEBUG = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if DEBUG:
    import numpy as np

    np.set_printoptions(suppress=True)


def plot_ssm(Rs, song_name, output_dir):
    for k, v in Rs.items():
        name = f"{song_name}-{k}"
        save_R_plot(v, name, output_dir)


def _get_song_name(fn: Path) -> str:
    return fn.with_suffix("").name


def process_song(
    songname: str,
    audio_file: Path,
    target_duration: int,
    weights: list[float] | None,
    output_dir: Path,
    tol_sec: int,  # default arg is in `run` since values are used there
    crossfade: float,
    *,
    plot: bool = True,
    similarity: Literal["cosine", "gaussian"] = "gaussian",
    full_measures: bool = False,
    use_seg_jump: bool = False,
) -> None:
    # Read song
    timer = Timer()
    timer.start("total_time")

    # AUDIO ANALYSIS
    sr = SAMPLE_RATE
    metadata = get_run_metadata(
        output_dir=output_dir,
        audio_file=audio_file,
        songname=songname,
        sr=sr,
        target_dur=target_duration,
        weights=weights,
        crossfade=crossfade,
        tolerance=tol_sec,
        similarity=similarity,
        full_measures=full_measures,
        seg_jump=use_seg_jump,
    )
    logger.info(f"Run metadata generated. metadata={metadata}")
    timer.start("analysis_time")
    waveform, beat_times, beat_numbers, R, Rs = analyze_song(
        str(audio_file),
        songname,
        sr,
        weights=weights,
        similarity=similarity,
    )

    num_beats = len(beat_times)
    bpm = average_bpm(beat_times)
    target_num_beats = seconds_to_beats(target_duration, bpm)
    timer.stop("analysis_time")

    # SEGMENTATION
    logger.info(f"Checking segment data cache. songname={songname}")
    seg = _check_seg_cache(songname)
    timer.start("segmentation_time")
    if seg:
        logger.info(f"Using segment data from cache. songname={songname}")
        segment_boundaries, segment_labels = seg
    else:
        logger.info(f"No segment data in cache. songname={songname}")
        save_beats_jams(beat_times, waveform, sr, audio_file)
        segment_boundaries, segment_labels = segment(audio_file)
        _cache_seg(songname, segment_boundaries, segment_labels)
    timer.stop("segmentation_time")

    if use_seg_jump:
        logger.info("Encouraging jumping at segment boundaries")
        segment_beats = segment_boundaries_to_beats(segment_boundaries, beat_times)
    else:
        segment_beats = None

    # GRAPH BUSINESS
    # Seconds tolerance in beats
    tolerance = seconds_to_beats(tol_sec, bpm)
    logger.info(
        f"Converting desired tolerance in seconds to beats. tolerance={tolerance},"
        f" tol_sec={tol_sec}, bpm={bpm}"
    )
    timer.start("search_time")
    G, btm = create_graph(
        R, target_num_beats, beat_numbers, tolerance, segment_beats, full_measures
    )

    beat_sequence = shortest_path(G, target_num_beats, num_beats, tolerance)
    timer.stop("search_time")
    if beat_sequence is None:
        logger.info("Could not find path")

        logger.info(f"Creating output directory for saving assets. output_dir={str(output_dir)}")
        # If it exists it was created this run so we can use it
        output_dir.mkdir(exist_ok=True)

        metadata["success"] = False
    else:
        logger.info(
            f"Received optimal beat sequence. beat_sequence={beat_sequence},"
            f" num_beats={beat_sequence.shape[0]}"
        )
        logger.info(
            f"Corresponding measure beat numbers. measure_beats={beat_numbers[beat_sequence]}"
        )

        # Convert beat times to waveform sample indices
        beat_samples = lr.time_to_samples(beat_times, sr=sr)
        sample_intervals = beats_to_intervals(beat_sequence, beat_samples, waveform)

        # OUTPUT ASSETS
        logger.info(f"Creating output directory for saving assets. output_dir={str(output_dir)}")
        # If it exists it was created this run so we can use it
        output_dir.mkdir(exist_ok=True)

        if plot:
            all_Rs = {"R": R}
            all_Rs.update(Rs)
            plot_ssm(all_Rs, songname, output_dir)

        remix_dur = render_remix(
            sample_intervals,
            waveform,
            sr,
            output_dir,
            songname,
            crossfade=crossfade,
            dur_n_tol=(target_duration, tol_sec),
        )
        timer.start("eval_time")
        scores = evaluate_remix(beat_sequence, beat_times, segment_boundaries, segment_labels)
        timer.stop("eval_time")
        logger.info(f"Calculated evaluation metrics. evaluation_scores={scores}")

        metadata["beat_sequence"] = beat_sequence.tolist()
        metadata["frame_intervals"] = sample_intervals.tolist()
        metadata["remix_duration"] = remix_dur
        metadata["remix_diff"] = abs(target_duration - remix_dur)
        metadata["evaluation_scores"] = scores
        metadata["success"] = True

    timer.stop("total_time")
    metadata["times"] = timer.get_times()
    save_metadata(metadata, output_dir)
    logger.info(f"Process completed in {timer.get_times()['total_time']}s.")


def get_output_dir(run_name: str) -> Path:
    output_dir = Path(OUTPUT_DIR) / run_name

    if output_dir.exists():
        raise FileExistsError(f"Output dir already exists: {output_dir}")

    logger.info(f"Output dir set: {output_dir}")
    return output_dir


def default_run_name(
    songname: str,
    duration: int,
    weights: list[float] | None,
    tol_sec: int,
    crossfade: float,
) -> str:
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    suffix = "".join(random.choice(string.ascii_letters) for i in range(3))
    if weights:
        w_str = "w" + "_".join(map(str, weights)) + "-"
    else:
        w_str = ""

    ts_suffix = f"{ts}-{songname}-{duration}s-{w_str}tol_{tol_sec}-cf_{crossfade}-{suffix}"
    return ts_suffix


def normalize_weights(weights: list[float]) -> list[float]:
    v = np.array(weights)
    if (v < 0).any():
        raise ValueError(f"Negative weight(s): {weights}")

    return (v / v.sum()).tolist()


def run(
    audio_file: Path,
    target_dur: int,
    weights: list[float] | None = None,
    tol_sec: int = DEFAULT_TOL_SEC,
    crossfade: float = DEFAULT_CROSSFADE,
    similarity: Literal["cosine", "gaussian"] = "gaussian",
    full_measures: bool = False,
    use_seg_jump: bool = False,
) -> None:
    """Prepare things around processing song request."""
    songname = _get_song_name(audio_file)
    logger.info(f"Lets go! {songname}")

    run_name = default_run_name(songname, target_dur, weights, tol_sec, crossfade)
    logger.info(f"Default run name set: {run_name}")

    output_dir = get_output_dir(run_name)

    if weights:
        weights = normalize_weights(weights)
        logger.info(f"Weights normalized to: {weights}")

    logger.info(f"Processing song: {audio_file}")
    process_song(
        songname,
        audio_file,
        target_dur,
        weights,
        output_dir,
        tol_sec,
        crossfade,
        similarity=similarity,
        full_measures=full_measures,
        use_seg_jump=use_seg_jump,
    )


def _file_arg(f) -> Path:
    p = Path(f)
    if not p.exists():
        raise FileNotFoundError(f"File {f} not found.")
    else:
        return p


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        prog="Sonata",
        description="Remix songs to match desired length. Assuming audio is 48kHz sample rate.",
        epilog="Cya!",
    )
    parser.add_argument("song", help="Song to be processed.", type=_file_arg)
    parser.add_argument("-d", "--target-dur", help="Desired length of remix.", default=15, type=int)
    parser.add_argument(
        "-w",
        "--weights",
        help=(
            "Weights of the features provided on the form x,y in order: MFCC, CQT."
            " They will be normalized."
        ),
        type=lambda s: [float(item) for item in s.split(",")],
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        help="Number of seconds the rendered remix can differ from the target duration.",
    )
    parser.add_argument("--cf", "--crossfade", type=float, help="Amount of crossfade. In seconds.")
    parser.add_argument("-s", "--similarity", choices=["gaussian", "cosine"])
    parser.add_argument(
        "--force-full",
        help="Force full measures. According to the beat tracker.",
        action="store_true",
    )
    parser.add_argument(
        "--use-seg-jump",
        help="Do not encourage jumping at segment borders.",
        action="store_true",
    )
    args = parser.parse_args()

    if args.weights and len(args.weights) != 3:
        raise ValueError(f"Bad weights: {args.weights}")

    logger.info(f"Received args: {args}")

    configuration = {
        "tol_sec": args.tolerance,
        "crossfade": args.cf,
        "similarity": args.similarity,
        "full_measures": args.force_full,
        "use_seg_jump": args.use_seg_jump,
    }

    # Filter None values
    filtered_conf = {k: v for k, v in configuration.items() if v is not None}
    logger.info(f"Filtered configuration. filtered_conf={filtered_conf}")

    run(args.song, args.target_dur, args.weights, **filtered_conf)
