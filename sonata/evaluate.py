import logging

import numpy as np

logger = logging.getLogger(__name__)


def section_recall(beat_sequence, beat_colors):
    """Consider the segmentation labels as ground truth for musical sections in
    the song.

    How many does the beat_sequence cover?
    Note: Only looking at the labels of the beats in the track (excluding intro/outro).
    """
    distinct_track_labels = set(beat_colors)
    sequence_colors = beat_colors[beat_sequence]
    distinct_sequence_colors = set(sequence_colors)

    logger.info(
        f"Computing music section recall. distinct_track_labels={distinct_track_labels},"
        f" distinct_sequence_labels={distinct_sequence_colors}"
    )

    return len(distinct_sequence_colors) / len(distinct_track_labels)


"""
Ideas
- weigh recall sections differently ,depending on portion of track size
    for instance bossalova had all sections but 0, which is by far the largest one

- How much of each part did it have?

Too advanced and possibly not desirable properties:
- Is the order of sections correct?
"""


def weighted_section_recall(beat_sequence, beat_colors):
    unique_labels, label_counts = np.unique(beat_colors, return_counts=True)

    label_sum = sum(label_counts)  # same as len(beat_colors)
    label_weights = label_counts / label_sum

    sequence_colors = beat_colors[beat_sequence]
    unique_sequence_colors = set(sequence_colors)

    weighted_recall = 0
    for i in sorted(unique_sequence_colors):
        # Get the
        weighted_recall += label_weights[np.where(unique_labels == i)[0][0]]

    return weighted_recall


def section_proportion_error(beat_sequence, beat_colors):
    """Not just ' is this section present' , but is the correct proporition of
    the section in the song?

    inspired by wenner

    if color 2 is 40% of the track (beat_sequence), then in the cutdown
    40% of the beats should be color 2.
    """
    logger.info("Calculating beat proportion error")
    distinct_track_labels, label_counts = np.unique(beat_colors, return_counts=True)

    label_sum = sum(label_counts)  # same as len(beat_colors)
    label_weights = label_counts / label_sum

    sequence_colors = beat_colors[beat_sequence]
    distinct_seq_colors, color_counts = np.unique(sequence_colors, return_counts=True)
    seq_len = len(beat_sequence)
    color_weights = color_counts / seq_len

    # No time for fancy vectorize np
    diffs = []
    sec_col_list = list(distinct_seq_colors)
    logger.warning(
        f"Temp info for debug: distinct_track_labels={distinct_track_labels},"
        f" label_counts={label_counts}, label_weights={label_weights},"
        f" beat_seq_color_weights={color_weights} see other log message for beat_seq colors"
    )
    for label, weight in zip(distinct_track_labels, label_weights):
        try:
            ix = sec_col_list.index(label)
            diffs.append(color_weights[ix] - weight)
        except ValueError:
            logger.warning(
                "Temp warning for seeing if there is a color missing from remix -"
                " check this track out!"
            )
            diffs.append(weight)

    logger.warn(f"Temp: diffs calculated: {diffs}")
    return np.sum(np.abs(diffs))


def assign_beat_colors(beat_times, boundaries, labels):
    beat_colors = np.zeros(len(beat_times), dtype=np.int32)
    # One more value in boundaries than labels becbeatause labels denotes ranges in boundaries
    for i in range(1, len(boundaries)):
        start_time = boundaries[i - 1]
        end_time = boundaries[i]
        color = labels[i - 1]
        beat_colors[(beat_times >= start_time) & (beat_times <= end_time)] = color

    return beat_colors


def count_backjumps(beat_sequence):
    backjumps = 0
    for i in range(1, len(beat_sequence)):
        if beat_sequence[i - 1] > beat_sequence[i]:
            backjumps += 1
    return backjumps


def track_coverage(beat_sequence, num_beats):
    """How many beats do we visit out of the total number of beats?

    Returns value in percent.
    """
    num_unique_beats = len(set(beat_sequence))
    return num_unique_beats / num_beats


def evaluate_remix(beat_sequence, beat_times, boundaries, labels):
    num_beats = len(beat_times)
    seq_length = len(beat_sequence)

    beat_colors = assign_beat_colors(beat_times, boundaries, labels)

    section_r = section_recall(beat_sequence, beat_colors)
    logger.info("Calculating weighted section recall")
    weighted_section_r = weighted_section_recall(beat_sequence, beat_colors)
    sec_prop_err = section_proportion_error(beat_sequence, beat_colors)

    logger.info("Counting back jumps")
    back_jumps = count_backjumps(beat_sequence)

    logger.info("Computing track coverage")
    track_cover_total = track_coverage(beat_sequence, num_beats)
    # How much of the track we could possible cover in seq_length many beats did we
    # actually cover. Perhaps more fair metric than _total.
    track_cover_possible = track_coverage(beat_sequence, seq_length)

    return {
        "section_recall": section_r,
        "weighted_section_recall": weighted_section_r,
        "sec_prop_err": sec_prop_err,
        "num_backjumps": back_jumps,
        "track_coverage_total": track_cover_total,
        "track_coverage_possible": track_cover_possible,
    }
