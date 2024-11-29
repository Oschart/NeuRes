import numpy as np
import math
from typing import List, Tuple

# TODO(dhs): duplication
def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

# TODO(dhs): duplication
def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

def reduce_res_tree(res_tree: list):
    # Res step format: ((c1, c2), c_res_idx, c_res)
    # Search from the end
    c_res = [r[2] for r in res_tree]
    try: falsum_idx = c_res.index(())
    except ValueError: return res_tree

    res_tree = res_tree[:falsum_idx+1]
    label = {}
    for r in res_tree:
        p = r[0]
        label[p[0]] = label[p[1]] = 0  
        label[r[1]] = 0
    label[res_tree[-1][1]] = 1
    for r in reversed(res_tree):
        p = r[0]
        label[p[0]] += label[r[1]]
        label[p[1]] += label[r[1]]
    red_tree = [r for r in res_tree if label[r[1]] > 0]
    return red_tree

def remove_idx_gaps(res_tree, min_idx):
    idx_remap = {}
    cont_tree = []
    for step in res_tree:
        if step[1] not in idx_remap:
            idx_remap[step[1]] = len(idx_remap) + min_idx
        c1 = idx_remap[step[0][0]] if step[0][0] in idx_remap else step[0][0]
        c2 = idx_remap[step[0][1]] if step[0][1] in idx_remap else step[0][1]
        new_t = tuple(sorted([c1, c2]))
        cont_tree.append((new_t, idx_remap[step[1]], step[2]))
    return cont_tree

class Formula(object):
    def __init__(self, n_vars, iclauses, is_sat, certificate):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.is_sat = is_sat

        if certificate is None:
            self.clauses = iclauses
        elif is_sat:
            # partial assignment for SAT
            self.preprocess_sat(iclauses, certificate)
        else:
            # Res proof for UNSAT
            self.preprocess_unsat(iclauses, certificate)
        
        self.n_clauses = len(self.clauses)
        self.n_cells = sum([len(clause) for clause in self.clauses])

        self.compute_L_unpack(self.clauses)

    def preprocess_unsat(self, clause_list, sol: List[str]):
        # Res step format: ((c1, c2), c_res_idx, c_res)
        formula = list(set(clause_list))
        clause_idx = {c:i for i, c in enumerate(formula)}
        clause_idx_orig = {c:i+1 for i, c in enumerate(clause_list)}
        idx_clause = {i+1:c for i, c in enumerate(clause_list)}

        ts = lambda s: tuple(sorted(s))
        def canon(x):
            # Note: this assumes binary res steps
            seq = tuple(map(int, x.split(' ')))
            clause = (ts(seq[-2:]), seq[0], ts(seq[1:-3]))
            return clause
        # Filter out formula restatement
        sol_steps = [canon(c) for c in sol if c[-2:] != ' 0']
        red_res = []  # reduced resolution steps
        # Shortcut using unit clauses + condense original clause refs
        for step in sol_steps:
            rp, c_idx, c = step[0], step[1], step[2]
            idx_clause[c_idx] = c
            if c not in clause_idx:
                red_res.append(step)
                clause_idx_orig[c] = c_idx
                clause_idx[c] = len(clause_idx)
            if len(c) == 1 and (-c[0],) in clause_idx:
                c_idx2 = clause_idx_orig[(-c[0],)]
                gen_pair = ts((c_idx, c_idx2))
                short_cut = (gen_pair, c_idx+1, ())
                red_res.append(short_cut)
                idx_clause[c_idx+1] = ()
                clause_idx_orig[()] = c_idx+1
                clause_idx[()] = len(clause_idx)
                break
        # Idx to idx mapping
        idx_map = {}
        for idx, c in idx_clause.items():
            assert c in clause_idx, "Mapping error"
            idx_map[idx] = clause_idx[c]
        # Apply idx mapping
        def remap_step(step):
            rp = ts([idx_map[x] for x in step[0]])
            c_idx = idx_map[step[1]]
            return (rp, c_idx, step[2])

        red_res1 = [remap_step(step) for step in red_res]
        red_res2 = self.remove_duplicate_steps(red_res1)
        red_res3 = reduce_res_tree(red_res2)
        red_res4 = remove_idx_gaps(red_res3, len(formula))

        self.clauses = formula
        self.certificate = red_res4
        return sol_steps

    def remove_duplicate_steps(self, res_tree):
        new_tree = []
        for step in res_tree:
            if step not in new_tree:
                new_tree.append(step)
        return new_tree

    def preprocess_sat(self, clause_list, sol: Tuple[int]):
        formula = list(set(clause_list))
        # VA = np.zeros(self.n_vars)
        VA = [0] * self.n_vars
        for v in sol:
            if v > 0:
                VA[abs(v)-1] = 1
            else:
                VA[abs(v)-1] = 0
        VA = tuple(VA)
        self.clauses = formula
        self.certificate = {VA}

    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = np.asarray([vlit, clause_idx])
                cell += 1

        assert(cell == self.n_cells)

    

def shift_ilit(x, offset):
    assert(x != 0)
    if x > 0: return x + offset
    else:     return x - offset

def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]

