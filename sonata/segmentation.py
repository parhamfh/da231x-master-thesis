import logging
import pickle
from pathlib import Path

import jams
import msaf
import numpy as np
import numpy.typing as npt
from msaf import input_output as io

SEG_DIR = "models/segmentation"
logger = logging.getLogger(__name__)


def _get_jams_path(audio_file: Path) -> str:
    jams_path = audio_file.parents[1] / "references" / audio_file.with_suffix(".jams").name

    return str(jams_path)


def save_beats_jams(beat_times, waveform, sr, audio_file: Path):
    """
    Source: https://jams.readthedocs.io/en/stable/examples.html

    Important that previous beat estimations do not shadow this jams.
    Delete `.features_msaf_tmp.json` in case there are issues of the type:
    ```
    Feature type FeatureTypes.ann_beatsync is not valid because no annotated beats were found
    ```
    """
    jams_path = _get_jams_path(audio_file)
    logger.info(f"Save beats calculated by BeatNet in JAMS format for MSAF. jams_path={jams_path}")

    jam = jams.JAMS()
    jam.file_metadata.duration = len(waveform) / sr
    beat_a = jams.Annotation(namespace="beat")
    beat_a.annotation_metadata = jams.AnnotationMetadata(data_source="BeatNet")

    #  Add beat timings to the annotation record.
    #  The beat namespace does not require value or confidence fields,
    #  so we can leave those blank.
    for t in beat_times:
        beat_a.append(time=t, duration=0.0)
    jam.annotations.append(beat_a)

    # NOTE: Just be sure you don't have a temporary features file in the directory.
    jam.save(jams_path)


def segment(audio_file: Path):
    algo = "scluster"
    logger.info(f"Calculated segments using MSAF. audio_file: {audio_file}, algorithm={algo}")
    bounds, labels = msaf.process(
        str(audio_file),
        feature="mfcc",
        boundaries_id=algo,
        labels_id=algo,
        annot_beats=True,
    )
    return bounds, labels


def hierarchical_segment(audio_file: Path):
    """Example."""
    conf = io.get_configuration(0, 0, 0, "scluster", None)
    conf["num_layers"] = 12
    bounds, labels = msaf.process(
        "audio/bossalova.wav",
        feature="mfcc",
        boundaries_id="scluster",
        hier=True,
        config=conf,
    )

    return bounds, labels


def segment_boundaries_to_beats(
    segment_boundaries: npt.NDArray[np.float32], beat_times: npt.NDArray[np.float32]
):
    # Some exceptional OCD checking
    if segment_boundaries[0] != 0:
        raise Exception("Expected first boundary to be 0")

    if segment_boundaries[1] != beat_times[0]:
        raise Exception("Expected first segment to be intro")

    # MSAF seems to use its own beat estimates for segments after the last
    # value of beat_times, we therefore only consider segments matching beat_times
    # since those segments are most likely silent outro according to BeatNet

    last_boundary_ix = max(
        [i for i, b in enumerate(map(lambda x: x in beat_times, segment_boundaries)) if b]
    )

    segment_beats = []
    start = np.where(beat_times == segment_boundaries[1])[0][0]
    for boundary_ix in range(2, last_boundary_ix + 1):
        end = np.where(beat_times == segment_boundaries[boundary_ix])[0][0]
        segment_beats.append((start, end))
        start = end
    # Add last segment and let it cover rest of track
    segment_beats.append((start, len(beat_times) - 1))

    # verify
    logger.info("Verify that segment ranges are correct")
    verification = [
        beat_times[segment_beats[i][0]] == segment_boundaries[i + 1]
        and beat_times[segment_beats[i][1]] == segment_boundaries[i + 2]
        for i in range(len(segment_beats))
    ]
    assert all(verification[:-1])
    # Assuming last beats will not match due to outro explanation above
    assert not verification[-1]

    return segment_beats


def _check_seg_cache(songname: str):
    cache_key = songname

    seg_path = Path(SEG_DIR) / f"{cache_key}.pickle"
    if seg_path.exists():
        logger.info(f"Loading cached segmentation data. cached_seg={seg_path}")
        with open(seg_path, "rb") as fp:
            boundaries, labels = pickle.load(fp)
        return boundaries, labels
    return None


def _cache_seg(songname: str, boundaries, labels):
    cache_key = songname

    seg_path = Path(SEG_DIR) / f"{cache_key}.pickle"
    logger.info(f"Saving MSAF segmentation. seg_path={seg_path}")
    data = (boundaries, labels)
    with open(seg_path, "wb") as fp:
        pickle.dump(data, fp)
