import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any


def save_metadata(run_data: dict, output_dir: Path):
    """Update json for run with `run_name` with values in data.

    Will create a JSON file for the run if it doesn't exist
    """
    name = run_data["run_name"]
    songname = run_data["songname"]
    with open(os.path.join(output_dir, f"{name}-{songname}.json"), "w") as fp:
        json.dump(run_data, fp)


def get_run_metadata(
    *,
    output_dir: Path,
    audio_file: Path,
    songname: str,
    sr: int,
    target_dur: int,
    weights: list[float] | None,
    tolerance: int,
    crossfade: float,
    similarity: str,
    full_measures: bool,
    seg_jump: bool,
) -> dict[str, Any]:
    run_name = output_dir.name
    return {
        "run_name": run_name,
        "audio_file": str(audio_file),
        "songname": songname,
        "sr": sr,
        "target_dur": target_dur,
        "weights": weights,
        "tolerance": tolerance,
        "crossfade": crossfade,
        "similarity": similarity,
        "full_measures": full_measures,
        "seg_jump": seg_jump,
    }


class Timer:
    times: dict
    start_times: dict

    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, timer_name: str):
        perf_counter()

        if timer_name in self.start_times:
            raise ValueError(f"Timer already exists with name: {timer_name}")

        self.start_times[timer_name] = perf_counter()

    def stop(self, timer_name: str):
        stop_time = perf_counter()

        self.times[timer_name] = stop_time - self.start_times[timer_name]

    def get_times(self):
        return self.times
