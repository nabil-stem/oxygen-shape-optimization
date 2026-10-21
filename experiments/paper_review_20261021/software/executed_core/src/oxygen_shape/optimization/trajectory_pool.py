"""Multiscale admissible proposals with hard resource checks before full validation.

Perimeter-preserving cell exchanges complement disk edits at a raster optimum.
They exchange exactly one cell and never repair or smooth the proposed mask.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.ndimage import convolve

from oxygen_shape.geometry.boundary import BoundaryKind, extract_boundary
from oxygen_shape.geometry.disk_edits import DiskEdit
from oxygen_shape.geometry.raster import OFFSETS
from oxygen_shape.optimization.feasible_pool import FeasiblePool
from oxygen_shape.optimization.guided_search import validate_candidate
from oxygen_shape.optimization.perimeter_pool import cheap_screen, draw_pool


@dataclass(frozen=True)
class CellFrontier:
    additions: list
    donors: dict


def cell_frontier(geometry, protected_length):
    mask = geometry.inside
    degree = convolve(
        mask.astype(np.int8), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode="constant", cval=0
    )
    additions, donors = [], {}
    faces = [
        f
        for f in extract_boundary(geometry)
        if f.kind == BoundaryKind.ROBIN
        and (f.key.col + 0.5) * geometry.h >= protected_length + 2 * geometry.h
    ]
    ny, nx = mask.shape
    for index, face in enumerate(faces):
        r, c = face.key.row, face.key.col
        k = int(degree[r, c])
        donors.setdefault(k, []).append((r, c, index))
        dr, dc = OFFSETS[face.key.side]
        rr, cc = r + dr, c + dc
        if (
            0 <= rr < ny
            and 0 <= cc < nx
            and not mask[rr, cc]
            and (cc + 0.5) * geometry.h >= protected_length + 2 * geometry.h
        ):
            additions.append((rr, cc, index, int(degree[rr, cc])))
    return CellFrontier([a for a in additions if a[3] in donors], donors)


def propose_cell_pair(geometry, frontier, rng):
    params = {
        "operation": "cell_pair",
        "radius": geometry.h / 2,
        "changed_cells": 1,
        "radius_meaning": "square_half_width",
        "addition_face": 0,
        "donor_face": 0,
    }
    if not frontier.additions:
        return DiskEdit(None, params, "No perimeter-preserving cell-pair frontier")
    ar, ac, ai, degree = frontier.additions[int(rng.integers(len(frontier.additions)))]
    donors = frontier.donors[degree]
    dr, dc, di = donors[int(rng.integers(len(donors)))]
    params.update(
        addition_face=ai,
        donor_face=di,
        add_row=ar,
        add_col=ac,
        donor_row=dr,
        donor_col=dc,
        add_x=(ac + 0.5) * geometry.h,
        add_y=(ar + 0.5) * geometry.h,
        remove_x=(dc + 0.5) * geometry.h,
        remove_y=(dr + 0.5) * geometry.h,
    )
    if abs(ar - dr) + abs(ac - dc) <= 1:
        return DiskEdit(None, params, "Cell exchange neighborhoods interact")
    mask = geometry.inside.copy()
    mask[ar, ac], mask[dr, dc] = True, False
    # Equal degrees k yield total exposed-face change (4-2k)+(2k-4)=0.
    return DiskEdit(geometry.with_mask(mask), params)


def build_trajectory_pool(
    config, geometry, initial, prepared, rng, remaining, seen, *, quota=2, initial_perimeter=None
):
    frontier = cell_frontier(geometry, config.protected_length)
    edits, rows, eligible = [], [], []
    signatures = {geometry.inside.tobytes(), *seen}
    checks, seconds = 0, 0.0
    for _ in range(min(remaining, config.refill_draw_limit)):
        proposal_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        local = np.random.default_rng(proposal_seed)
        began = perf_counter()
        if local.random() < config.cell_move_probability:
            edit = propose_cell_pair(geometry, frontier, local)
            row = dict(edit.parameters)
        else:
            batch, metadata = draw_pool(geometry, local, config, prepared, 1)
            edit, row = batch[0], metadata[0]
        row.update(proposal_seed=proposal_seed, generation_seconds=perf_counter() - began)
        reason, perimeter = cheap_screen(edit, initial, config)
        if not reason and config.perimeter_band is not None:
            if abs(perimeter - initial_perimeter) > config.perimeter_band + 1e-12:
                reason = "Exact-perimeter control rejected before full feature checks"
        checked = False
        if not reason:
            signature = edit.geometry.inside.tobytes()
            if signature in signatures:
                reason = "Duplicate or incumbent mask"
            else:
                began = perf_counter()
                reason = validate_candidate(edit.geometry, initial, config)
                seconds += perf_counter() - began
                checks += 1
                checked = True
                signatures.add(signature)
        edits.append(edit)
        eligible.append(not reason)
        rows.append(
            {
                **row,
                "slot": len(edits) - 1,
                "screen_rejection": reason,
                "perimeter": perimeter,
                "full_checked": checked,
            }
        )
        if len(edits) >= config.pool_size and sum(eligible) >= quota:
            break
    return FeasiblePool(
        edits, rows, np.asarray(eligible, bool), checks, seconds, sum(eligible) >= quota
    )
