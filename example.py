import logging
import secrets
import string
from pathlib import Path

import librosa as lr

from sonata.analyze import analyze_song, average_bpm, seconds_to_beats
from sonata.graph import create_graph, shortest_path
from sonata.render import beats_to_intervals, render_remix

# TODO: set level debug in app
OUTPUT_DIR = "output"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEBUG = True
if DEBUG:
    import numpy as np

    np.set_printoptions(suppress=True)


def get_output_dir(song_name: str) -> Path:
    suffix = "".join(secrets.choice(string.ascii_letters) for i in range(3))
    output_dir = Path(OUTPUT_DIR) / f"{song_name}-{suffix}"
    # RAises if exists
    output_dir.mkdir()
    return output_dir


def plot_ssm(Rs, output_dir):
    for k, v in Rs.items():
        # from datetime import datetime
        from sonata.plots import save_R_plot

        # ts_suffix = datetime.now().strftime("%y%m%d-%H%M%S")
        save_R_plot(v, k, output_dir)


def demo(target_duration: int = 15, plot: bool = True) -> None:
    # TODO: take path to audio, get song name from filepath
    # Read song
    song_name = "dezmoran"
    # song_name = "pineapple"
    logger.info(f"Lets go! {song_name}")

    output_dir = get_output_dir(song_name + "-improve")
    logger.info(f"Storing all output in: {output_dir}")

    # Load audio
    sr = 48000
    waveform, beat_times, beat_numbers, R, Rs = analyze_song(
        f"audio/{song_name}.wav", song_name, sr=sr
    )

    if plot:
        plot_ssm(Rs, output_dir)

    num_beats = len(beat_times)
    bpm = average_bpm(beat_times)
    target_num_beats = seconds_to_beats(target_duration, bpm)

    # TODO: Segment

    # Seconds tolerance in beats
    tol_sec = 4
    tolerance = seconds_to_beats(tol_sec, bpm)
    G, btm = create_graph(R, target_num_beats, beat_numbers, tolerance, None, False)

    # TODO: Graph search
    beat_sequence = shortest_path(G, target_num_beats, num_beats, tolerance)

    if beat_sequence is None:
        logger.info("Could not find path")
        return
    logger.info(
        f"Received optimal beat sequence. beat_sequence={beat_sequence},"
        f" num_beats={beat_sequence.shape[0]}"
    )
    logger.info(f"Corresponding measure beat numbers. measure_beats={beat_numbers[beat_sequence]}")

    # Convert beat times to waveform sample indices
    beat_samples = lr.time_to_samples(beat_times, sr=sr)
    sample_intervals = beats_to_intervals(beat_sequence, beat_samples, waveform)

    render_remix(
        sample_intervals,
        waveform,
        sr,
        output_dir,
        song_name,
        dur_n_tol=(target_duration, tol_sec),
    )


if __name__ == "__main__":
    demo()
