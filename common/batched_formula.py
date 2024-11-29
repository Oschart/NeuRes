import torch as th
import numpy as np
from dataclasses import dataclass
from typing import List
from common.problem import Formula


@dataclass
class BatchedFormula:
    """Padded snapshot of B formulas for batched message-passing.

    Mask convention: True = valid (real literal/var/clause) everywhere.
    L_unpack and clause_mask are pre-allocated to H_max along the clause
    dimension; the embedder mutates them in place as new clauses are added,
    so consumers must treat them as live references rather than snapshots.
    """
    formulas: List[Formula]
    n_vars: th.LongTensor       # (B,)
    n_lits: th.LongTensor       # (B,)  = 2 * n_vars
    n_clauses: th.LongTensor    # (B,)  initial clause counts
    max_vars: int
    max_lits: int               # = 2 * max_vars
    max_clauses: int            # initial; H_max is L_unpack.shape[-1]
    H_max: int                  # capacity along clause dim (initial + horizon)
    L_unpack: th.Tensor         # (B, max_lits, H_max) float, 0/1 dense; mutable
    lit_mask: th.BoolTensor     # (B, max_lits)
    var_mask: th.BoolTensor     # (B, max_vars)
    clause_mask: th.BoolTensor  # (B, H_max); True only at occupied slots; mutable
    flip_idx: th.LongTensor     # (B, max_lits) — pos↔neg swap; padding maps to self
    device: th.device

    @property
    def B(self) -> int:
        return len(self.formulas)


def _make_flip_idx(n_vars_per_row: List[int], max_vars: int, device) -> th.LongTensor:
    """In the padded layout, positive literals live at [0, max_vars) and negative
    at [max_vars, 2*max_vars). For each row b with n_vars=V_b, positions
    0..V_b-1 swap with max_vars..max_vars+V_b-1; all padding positions
    (V_b..max_vars-1 and max_vars+V_b..2*max_vars-1) map to themselves."""
    B = len(n_vars_per_row)
    max_lits = 2 * max_vars
    flip = th.arange(max_lits, device=device).unsqueeze(0).repeat(B, 1).contiguous()
    for b, V in enumerate(n_vars_per_row):
        pos_range = th.arange(V, device=device)
        flip[b, :V] = max_vars + pos_range
        flip[b, max_vars:max_vars + V] = pos_range
    return flip


def batch_formulas(formulas: List[Formula], device, H_max: int) -> BatchedFormula:
    """Build a BatchedFormula snapshot from a list of Formulas.

    H_max must be at least max_b(formula.n_clauses + horizon_b); the embedder
    writes new clauses into L_unpack[:, :, n_clauses_b:H_max] in-place during
    the episode, so this dictates upfront memory.
    """
    B = len(formulas)
    n_vars_list = [f.n_vars for f in formulas]
    n_lits_list = [f.n_lits for f in formulas]
    n_clauses_list = [f.n_clauses for f in formulas]

    max_vars = max(n_vars_list)
    max_lits = 2 * max_vars
    max_clauses = max(n_clauses_list)
    assert H_max >= max_clauses, f"H_max={H_max} must be >= max initial n_clauses={max_clauses}"

    # Initial L_unpack scattered into pre-allocated (B, max_lits, H_max) dense tensor.
    # Per-formula vlits use n_vars_b as the pos/neg boundary (see Formula.compute_L_unpack);
    # we remap negatives to the padded layout where the boundary is max_vars,
    # so flip_idx (which assumes [0, max_vars)=pos, [max_vars, 2*max_vars)=neg) works.
    L_unpack = th.zeros(B, max_lits, H_max, device=device)
    for b, f in enumerate(formulas):
        if f.n_cells == 0:
            continue
        idx_np = f.L_unpack_indices.copy()  # (n_cells, 2): [vlit, clause_idx]
        neg = idx_np[:, 0] >= f.n_vars
        idx_np[neg, 0] = idx_np[neg, 0] - f.n_vars + max_vars
        idx = th.from_numpy(idx_np).long()
        L_unpack[b, idx[:, 0], idx[:, 1]] = 1.0

    # Masks
    lit_mask = th.zeros(B, max_lits, dtype=th.bool, device=device)
    var_mask = th.zeros(B, max_vars, dtype=th.bool, device=device)
    clause_mask = th.zeros(B, H_max, dtype=th.bool, device=device)
    for b, f in enumerate(formulas):
        var_mask[b, :f.n_vars] = True
        # Real literals occupy [0, n_vars) and [max_vars, max_vars+n_vars). The
        # second block lives in the padded region, so we mark both halves
        # explicitly to keep flip()'s pos↔neg pairing valid post-padding.
        lit_mask[b, :f.n_vars] = True
        lit_mask[b, max_vars:max_vars + f.n_vars] = True
        clause_mask[b, :f.n_clauses] = True

    flip_idx = _make_flip_idx(n_vars_list, max_vars, device)

    return BatchedFormula(
        formulas=formulas,
        n_vars=th.tensor(n_vars_list, dtype=th.long, device=device),
        n_lits=th.tensor(n_lits_list, dtype=th.long, device=device),
        n_clauses=th.tensor(n_clauses_list, dtype=th.long, device=device),
        max_vars=max_vars,
        max_lits=max_lits,
        max_clauses=max_clauses,
        H_max=H_max,
        L_unpack=L_unpack,
        lit_mask=lit_mask,
        var_mask=var_mask,
        clause_mask=clause_mask,
        flip_idx=flip_idx,
        device=device,
    )
