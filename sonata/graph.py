import logging
from itertools import product

import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy.sparse import bsr_matrix

logger = logging.getLogger(__name__)
# Cost of paths we want to block
BLOCK_COST = 100


def create_btm(ssm):
    """BTM according to Stoller's paper."""
    btm = np.roll(ssm, -1, axis=0)
    # Due to illeguralities
    btm[btm > 1] = 1

    # Think it is ok because they exist in G in `shortest_path` if you print it
    btm = 1 - btm
    return btm


def create_G(btm, target_num_beats, tolerance):
    # num_beats = len(btm)
    max_path_length = target_num_beats + tolerance
    logger.info(
        f"Creating beat graph from beat transition matrix. target_num_beats={target_num_beats},"
        f" tolerance_beats={tolerance}, max_path_length={max_path_length}"
    )

    # Add data index axis so shape becomes (1, num_beats, num_beats)
    graph_data = np.expand_dims(btm, 0)
    graph_data = np.repeat(graph_data, max_path_length, axis=0)

    # block column indices
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html#scipy.sparse.bsr_matrix
    # +1 because nodes of beat i transition to nodes of beat j
    indices = np.arange(max_path_length) + 1
    indices[-1] = 0  # Last beat will be ignored but for safety
    indptr = np.arange(max_path_length + 1)
    indptr[-1] = indptr[-2]  # Zeros out last beat's rows

    G = bsr_matrix((graph_data, indices, indptr))

    return G


def stay_on_meter(btm, beat_numbers, full_measures=False):
    """Assume btm and beat_numbers same length.

    Beat_numbers is assumed to be 1-indexed according to BeatNet. Does
    not handle if time signature changes in song. btm contains
    transition costs. Adds 100 to transitions of meter.

    full_measures   - only allow transitioning after a measure has ended. This means
                      you an only jump from the final beat of a measure.
                      Ex: beat 4s can go to other 1s, but no other beat number may jump.
    """
    logger.info(f"Rule: Forcing beats to adhere to time signature. full_measures={full_measures}")
    assert len(btm) == len(beat_numbers)  # assume nothing!!!
    num_beats = len(beat_numbers)

    # Assume same meter in entire track. This breaks if different measures
    measure_length = int(max(beat_numbers))

    # Calculate penalities for each beat position
    if full_measures:
        mask_matrix = np.full((measure_length, num_beats), BLOCK_COST)

        # Allow jumps from last beat of measure
        mask = np.zeros(num_beats)
        i = measure_length
        # What are indices of beats in same position in measure as next.
        mask[beat_numbers != (i % measure_length) + 1] = BLOCK_COST
        # Zero-indexed
        mask_matrix[i - 1] = mask
    else:
        beat_masks = []
        for i in range(1, measure_length + 1):
            mask = np.zeros(num_beats)
            # What are indices of beats in same position in measure as next. 1-indexed
            mask[beat_numbers != (i % measure_length) + 1] = BLOCK_COST
            beat_masks.append(mask)

        mask_matrix = np.array(beat_masks)

    # CREATE MASK
    # shift to song's start beat
    first_beat_num = int(beat_numbers[0])
    shift = 1 - (first_beat_num)
    mask_matrix = np.roll(mask_matrix, shift, axis=0)

    # Repeat and ensure covers BTM, repeat value ensures there are more rows than we need
    btm_mask = np.tile(mask_matrix, (num_beats // (measure_length - 1), 1))[:num_beats, :]
    if full_measures:
        # Guarantee actual next beat is not modified
        next_diag = np.roll(np.eye(num_beats), 1, axis=1)
        btm_mask[next_diag == 1] = 0

    # Ensure we don't jump from last beat
    btm_mask[-1] = BLOCK_COST

    # Apply mask
    return btm + btm_mask


def _get_penalty_matrix(size: int, w: int = 4) -> npt.NDArray[np.int32]:
    """Like Stoller 4.1.2 (1)"""
    mx = np.atleast_2d(np.arange(size, dtype=np.float32))
    mx = np.repeat(mx, size, axis=0)

    # k = -1 zero out main diagonal
    p_mx = np.tril((mx.T + 1) - mx, k=-1)
    # To avoid x/0 => inf

    c_rep = np.divide(w, p_mx, out=np.zeros_like(p_mx), where=p_mx != 0)

    # at most penalize with -1, 4.1.2 (3)
    c_rep = np.minimum(c_rep, 1, where=c_rep != 0)
    return c_rep


def add_back_jump_penalty(btm):
    logger.info("Rule: Adding penalty for jumping back")
    num_beats = btm.shape[0]
    p_mx = _get_penalty_matrix(num_beats)
    return btm + p_mx


def encourage_segments(btm, segment_beats):
    """Operates in place."""
    logger.info("Rule: Encouraging jumping at segment boundaries")
    # Valid segment jump points
    segment_starts = [sb[0] for sb in segment_beats[1:]]
    segment_ends = [sb[1] for sb in segment_beats[:-1]]

    # Get all pair of jump from->to beat indices except those jumping to self
    candidates = list(filter(lambda x: x[0] != x[1], product(segment_starts, segment_ends)))

    discount = 0.3
    logger.info(
        f"Segment candidates to receive jump discount. num_candidates={len(candidates)},"
        f" discount={discount}"
    )
    for can in candidates:
        btm[can[0], can[1]] = btm[can[0], can[1]] * discount
        # Sigmoud discount: larger if more similar
        # cost = btm[can[0], can[1]]
        # btm[can[0], can[1]] = cost * np.minimum(1/(1+np.exp(-cost)), 1)


def apply_rules(btm, beat_numbers, segment_beats, full_measures):
    """Rule Rules.

    Order of applying rules matter.

    1. Stay on meter zeros out illegal moves: = 0
    2. Segment jumps encourarages jumps: cost * factor
    3. Jump penalty adds penalty according to distance should not be included in
       step 2: cost + penalty
    """
    logger.info(
        f"Applying rules to transitions. full_measures={full_measures}"
        f" segment_beats={bool(segment_beats)}"
    )

    # Assumes same time signature in entire song
    metred_btm = stay_on_meter(btm, beat_numbers, full_measures=full_measures)

    # Encourage segment jumps
    if segment_beats:
        # Modifies in-place
        encourage_segments(metred_btm, segment_beats)

    # This breaks BTM symmetry i->j and j->i might not have same cost. Which is correct.
    penalty_btm = add_back_jump_penalty(metred_btm)

    return penalty_btm


def create_graph(ssm, target_num_beats, beat_numbers, tolerance, segment_beats, full_measures):
    # Create beat transition matrix
    logger.info("Creating graph from self-similarity matrix")
    btm = create_btm(ssm)

    btm = apply_rules(btm, beat_numbers, segment_beats, full_measures)

    # Compile graph for desired path length
    G = create_G(btm, target_num_beats, tolerance)

    return G, btm


def shortest_path(
    G: sp.sparse.bsr_matrix, target_num_beats: int, num_beats: int, tolerance: int
) -> npt.NDArray[np.int32] | None:
    """
    G                   - prepared graph for beat transitions
    target_num_beats    - desired length of graph path
    num_beats           - total number of beats in song
    tolerance           - acceptable deviation from target_num_beats. G must be large
                        enough to cover `target_num_beats + tolerance`
    """
    logger.info(
        f"Searching for shortest path in graph. target_num_beats={target_num_beats},"
        f" tolerance_beats={tolerance}"
    )
    costs, predecessors = sp.sparse.csgraph.shortest_path(
        G, method="D", indices=0, return_predecessors=True
    )

    target_end_beat = num_beats - 1
    potential_targets = [
        (target_num_beats + x) * num_beats + target_end_beat
        # We accept beats before final
        for x in range(-tolerance, tolerance)
    ]
    logger.info(
        f"target_number_of_beats: {target_num_beats}\n num_beats: {num_beats}\n"
        f" target_end_beat: {target_end_beat}"
    )
    logger.info(f"Potential targets: {potential_targets}")
    logger.info(f"Potential targets in orig form: {[p % num_beats for p in potential_targets]}")

    paths = []

    for target in potential_targets:
        if predecessors[target] == -9999:
            continue
        path = [np.int32(target)]
        # initial cutdown beat is 0
        while path[-1] != 0:
            path.append(predecessors[path[-1]])
        path = path[::-1]
        paths.append(path)
    if len(paths) == 0:
        logger.info("No path found.")
        return None

    path_costs = [costs[p[-1]] for p in paths]

    # Length-normalize path costs.
    for i, (p, c) in enumerate(zip(paths, path_costs)):
        path_costs[i] = c / (len(p) - 1)

    graph_path = paths[np.argmin(path_costs)]

    beat_sequence = np.asarray(graph_path) % num_beats
    return beat_sequence
