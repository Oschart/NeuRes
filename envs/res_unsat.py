import torch as th
import numpy as np
import random
import pandas as pd
from copy import deepcopy
from scipy import sparse
from typing import Any, Optional, Dict, List, Tuple
from common.problem import Formula
from common.batched_formula import BatchedFormula, batch_formulas
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from common.utils import radix_comp


class ResUNSAT():
    def __init__(self, config, test_ratio=0.1, shuffle=True, overfit=False):
        super(ResUNSAT, self).__init__()
        self.config = config
        self.mode = "train"
        self.variant = config['dataset']
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.overfit = overfit
        self.supervised = False
        self.reward_map = {
            "binary": self.binary_R,
            "length": self.length_R,
            "lookahead": self.lookahead_R,
        }

        self.dataset = None
        self.stats = {"train": {}, "val": {}, "test": {}}
        self.iter_state = None
        self.state = None
        self.sample_idx = 0
        self.iter_idx = None
        self.sample_idxs = None
        self.n_chunks = config['n_chunks']
        self.chunk_id = config['chunk_id']
        self.st_idx = 0
        self.end_idx = None
        

        self.formula: Formula = None
        self.res_trail = []
        self.init_dataset()
        self._setup_guide()
    
    def set_mode(self, mode: str):
        assert mode in ["train", "val", "test"], "Mode must be either train, val, or test"
        self.mode = mode

        pool_size = len(self.dataset[self.mode]["problems"])
        chunk_size = pool_size//self.n_chunks
        self.st_idx = chunk_size * self.chunk_id
        if self.chunk_id == self.n_chunks - 1:
            chunk_size += pool_size%self.n_chunks
        self.end_idx = self.st_idx+chunk_size
        # self.sample_idxs = list(range(len(self.dataset[self.mode]["problems"])))
        self.sample_idxs = list(range(self.st_idx, self.end_idx))
        print(f"set_mode: Processing from {self.st_idx} --> {self.end_idx}")
    
    def init_dataset(self):
        if self.config['test_only']:
            pickled_path = f"data/{self.variant}/test_dataset.pkl"
        else:
            pickled_path = f"data/{self.variant}/dataset.pkl"
        if os.path.exists(pickled_path):
            dataset_dict = pkl.load(open(pickled_path, "rb"))
        else:
            dataset_dict = self.load_dataset()
            pkl.dump(dataset_dict, open(pickled_path, "wb"))
    
        if self.overfit:
            # Pick the first UNSAT instance from train so the supervised loss
            # has a non-trivial resolution proof to fit. Datasets like rcar-1
            # mix SAT and UNSAT, so a hardcoded index can land on the wrong type.
            train_problems = dataset_dict["train"]["problems"]
            try:
                sample = next(p for p in train_problems if not p.is_sat)
            except StopIteration:
                raise RuntimeError(
                    f"Overfit mode requires at least one UNSAT instance in "
                    f"data/{self.variant}/train, but none were found."
                )
            for mode in ["train", "val", "test"]:
                if mode not in dataset_dict:
                    dataset_dict[mode] = {}
                dataset_dict[mode]["problems"] = [sample]

        if self.config['test_only']:
            dataset_dict = {"test": dataset_dict["test"]}

        self.dataset = dataset_dict
        # UNSAT size stats
        if self.config['test_only']:
            self.orig_res_size = sum([len(s.certificate) for s in dataset_dict["test"]["problems"] if not s.is_sat])
            self.total_problems = len(dataset_dict["test"]["problems"])
        else:
            self.orig_res_size = sum([len(s.certificate) for s in dataset_dict["train"]["problems"] if not s.is_sat])
            self.total_problems = len(dataset_dict["train"]["problems"])
        self.shrunk_res_size = self.orig_res_size
        
        self.reproven = 0
        # SAT size stats
        self.sat_history = {}


    def load_dataset(self):
        dataset_dict = {}
        for mode in ["train", "val", "test"]:
            total_sat = 0
            bad_sat = 0
            print(f"Loading {mode} dataset...")
            problems = []
            if not os.path.exists(f"data/{self.variant}/{mode}.csv"):
                continue

            df = pd.read_csv(f"data/{self.variant}/{mode}.csv")
            str_problems = df["formula"].to_list()
            if "sat" in df:
                sat_status = df["sat"].to_list()
            else:
                sat_status = [False] * len(str_problems)
            if "assignment" in df:
                str_assignments = df["assignment"].to_list()
            else:
                str_assignments = None
            str_res_proofs = df["res_proof"].to_list()
            
            n_problems = len(str_problems)
            for i in tqdm(range(n_problems)):
                p = str_problems[i].split(" 0\\n")[:-1]
                specs, p[0] = p[0].split("\\n")
                is_sat = bool(sat_status[i])
                if self.config['sat_only'] and not is_sat:
                    continue
                if is_sat:
                    sol = tuple(map(int, str_assignments[i].split(",")))
                else:
                    sol = str_res_proofs[i].split(" 0\\n")[:-1]

                n_vars, n_clauses = list(map(int, specs.split(' ')[-2:]))
                clauses = [tuple(sorted(map(int, c.split(' ')))) for c in p]
                
                problem = Formula(n_vars, clauses, is_sat, sol)
                problems.append(problem)
                if is_sat:
                    total_sat += 1
                    bad_sat += int(not self.verify_sat_cert(problem))

            
            dataset_dict[mode] = {"problems": problems}
            print(f"Bad {mode} SAT proofs: {bad_sat}/{total_sat}")
        if self.config['test_split'] > 0.0:
            train_problems = dataset_dict["train"]["problems"]
            train_problems, val_problems = train_test_split(train_problems, test_size=self.config['test_split'])
            dataset_dict["train"]["problems"] = train_problems
            dataset_dict["val"] = {"problems": val_problems}
        return dataset_dict
    
    def save_data_stats(self, model_name):
        save_dir = f"data/{self.variant}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        pkl.dump(self.stats, open(f"{save_dir}/shrunk_stats.pkl", "wb"))
        pkl.dump(self.dataset, open(f"{save_dir}/dataset.pkl", "wb"))

    

    def _setup_guide(self):
        def supervised_step(x):
            if self.is_sat:
                while True:
                    yield {"VA": self.solution, "Res": None}
            else:
                derived_map = {s[1]: s[2] for s in self.solution}
                cform = lambda c: derived_map[c] if c in derived_map else self.clauses[c]
                for step in self.solution:
                    c1, c2 = cform(step[0][0]), cform(step[0][1])
                    c_idx = (self.clause_map[c1], self.clause_map[c2])
                    yield {"VA": None, "Res": c_idx}
        def unsupervised_step(x): 
            while True: yield {"VA": None, "Res": None}
        
        if self.supervised: guide = supervised_step(None)
        else: guide = unsupervised_step(None)
        self.guide = guide
        
    def reset(self):
        sample_idx = random.randint(0, len(self.dataset[self.mode]["problems"])-1)#1217#
        self.sample_idx = sample_idx

        self.init_episode(sample_idx)
        return self.formula
    
    def save_iter_state(self):
        self.iter_state = {
            "sample_idx": self.sample_idx,
            "iter_idx": self.iter_idx,
            "sample_idxs": self.sample_idxs,
            "mode": self.mode,
            "st_idx": self.st_idx,
            "end_idx": self.end_idx,
            "supervised": self.supervised,
        }
    
    def restore_iter_state(self):
        self.sample_idx = self.iter_state["sample_idx"]
        self.iter_idx = self.iter_state["iter_idx"]
        self.sample_idxs = self.iter_state["sample_idxs"]
        self.mode = self.iter_state["mode"]
        self.st_idx = self.iter_state["st_idx"]
        self.end_idx = self.iter_state["end_idx"]
        self.supervised = self.iter_state["supervised"]

    
    def __iter__(self):
        self.iter_idx = 0
        # self.sample_idxs = list(range(len(self.dataset[self.mode]["problems"])))[st_idx:end_idx]
        self.sample_idxs = list(range(self.st_idx, self.end_idx))
        print(f"#Samples = {len(self.sample_idxs)}")
        print(f"__iter__: Processing from {self.st_idx} --> {self.end_idx}")
        if self.shuffle:
            random.shuffle(self.sample_idxs)
        return self
    
    def __next__(self):
        if self.iter_idx >= len(self.sample_idxs):
            raise StopIteration
        self.sample_idx = self.sample_idxs[self.iter_idx]
        self.init_episode(self.sample_idx)
        self.iter_idx += 1
        return self.formula

    def __len__(self):
        return len(self.sample_idxs)

    def init_episode(self, sample_idx=None):
        if sample_idx is None:
            sample_idx = self.sample_idx
        self.formula = self.dataset[self.mode]["problems"][sample_idx]
        self.solution = self.formula.certificate
        self.is_sat = self.formula.is_sat
        
        if self.is_sat:
            optim_factor = 2 if self.mode == "train" else 4
            self.H = self.formula.n_vars*optim_factor
            if self.config['max_H'] is not None:
                self.H = self.config['max_H']
        else:
            optim_factor = 3 if self.mode == "train" else 4
            if self.config['max_pLen'] is not None:
                optim_factor = self.config['max_pLen']
            self.H = len(self.solution)*optim_factor
        
        

        self.clauses = self.formula.clauses.copy()
        self.clause_map = {c: i for i, c in enumerate(self.clauses)}
        self.build_clause_mat()
        self.build_res_mask()
        # self.get_res_mask()
        self.res_trail = []
        self.min_len = self.formula.n_vars # np.mean(list(map(len, self.clauses)))
        self._setup_reward()
        self._setup_guide()
        self.timeout_count = self.H
        self.found_sat_VA = False
        self.sat_VA_parity = -1
        if self.config['track_VAs']:
            self.VAs = set()
            self.unique_VAs = np.zeros(self.H)
        return self.formula
    
    
    def guide_step(self):
        return next(self.guide)
    
    def binary_R(self):
        def R(c_new):
            if c_new is None: return 0
            return int(len(c_new) == 0)
        return R
    
    def length_R(self):
        '''
            Assign exponentially increasing rewards based on level (clause length)
            Level0 is the empty clause and has the highest reward (=1)
            Once a level has been hit, the solver no longer takes reward for it.
        '''
        n_levels = self.formula.n_vars
        self.r_levels = np.logspace(0, n_levels, n_levels, base=2)/2**(n_levels)
        self.r_levels = np.flip(self.r_levels)
        self.r_visited = [False] * n_levels
        def R(c_new):
            if c_new is None: return 0
            c_len = len(c_new)
            reward = (not self.r_visited[c_len])*self.r_levels[c_len]
            self.r_visited[c_len] = True
            return reward
        return R
    
    def lookahead_R(self):
        n_levels = self.formula.n_vars
        self.r_levels = np.logspace(0, n_levels, n_levels, base=2)/2**(n_levels)
        self.r_levels = np.flip(self.r_levels)
        self.leaders = [None] * n_levels
        self.build_lookahead_mat()
        def R(c_new):
            if c_new is None: return 0
            self.update_lookahead_mat()
            curr_score = self.S1[-1]
            leader = self.leaders[len(c_new)]
            if leader is None:
                leader_score = np.zeros_like(curr_score)
            else:
                leader_score = self.S1[leader]
            r_comp = radix_comp(curr_score, leader_score)
            if r_comp == 1:
                if len(c_new) > 0: self.r_levels[len(c_new)] /= 2
                self.leaders[len(c_new)] = len(self.clauses)-1
                return self.r_levels[len(c_new)]
            return 0
        return R

    def _setup_reward(self):
        if self.mode == "val":
            self.R = self.binary_R()
        else:
            self.R = self.reward_map[self.config['reward']]()

    # @profile
    def step(self, action: dict) -> np.array:
        '''
            Takes indices to two clauses to perform resolution on
            and returns the resultant clause
        '''
        self.timeout_count -= 1

        Res = action["Res"] if "Res" in action else None
        VAs = action["VA"] if "VA" in action else None
        # Derive resultant clause
        if Res is not None:
            c_new = []
            for r in Res:
                c1, c2 = self.clauses[r[0]], self.clauses[r[1]]
                self.res_mask[r[0], r[1]] = False
                self.res_mask[r[1], r[0]] = False
                new_clause = self.resolve(c1, c2)
                if new_clause is not None:
                    self.res_trail.append((r, len(self.clauses), new_clause))
                    c_new.append(new_clause)
                    self.add_clause(new_clause)
        
            # This is a symbolic clause, merely for utility, it's not a Markovian observation
            next_obs = c_new 
            # Incentivize shorter proof
            reward = 0 #self.R(c_new)
        else:
            c_new = None
            next_obs = None
            reward = 0
        # We're done when we derive the empty clause
        # proven_unsat = c_new is not None and len(c_new) == 0
        proven_unsat = c_new is not None and any([c is not None and len(c) == 0 for c in c_new])
        if self.found_sat_VA:
            proven_sat = True
        else:
            proven_sat = self.is_sat and self.is_satisfying(VAs)
            if self.config['track_VAs']:
                if VAs is not None and VAs[0] not in self.VAs:
                    self.VAs.add(VAs[0])
                    self.unique_VAs[-self.timeout_count - 1] += 1
                    
        solved = proven_unsat or proven_sat
        timeout = self.timeout_count <= 0
        if self.config['penalize_timeout'] and timeout:
            reward = -1
            
        done = solved or timeout

        # No special info to send back
        info = {"solved": solved}
        
        return next_obs, reward, done, info

    def is_satisfying(self, VAs):
        if VAs is None: return False
        N = self.formula.n_clauses
        orig_mat = self.clause_mat[:N]
        for i, VA in enumerate(VAs):
            if isinstance(VA, th.Tensor):
                VA = VA.squeeze(0).detach().cpu().numpy()
            # Check if VA satisfies the original clauses
            VA_sign = VA.copy()
            VA_sign[VA_sign == 0] = -1
            sub_mat = orig_mat*VA_sign
            satisfied = np.any(sub_mat == 1, axis=-1)
            sat = np.all(satisfied)
            if sat:
                if self.config['adapt_sat_gt']:
                    self.register_VA(VA)
                self.sat_VA_parity = i
                return True
        return False
    
    def verify_sat_cert(self, formula: Formula):
        mat = np.zeros((formula.n_clauses, formula.n_vars), dtype=int)
        for i, c in enumerate(formula.clauses):
            vec = np.zeros(formula.n_vars)
            if len(c) == 0: return vec
            vec[np.abs(c) - 1] = np.sign(c)
            mat[i] = vec
        for VA in formula.certificate:
            VA = np.array(VA)
            VA[VA == 0] = -1
            N = formula.n_clauses
            orig_mat = mat[:N]
            sub_mat = orig_mat*VA
            sat = np.all(np.any(sub_mat == 1, axis=-1))
            return sat
    
    def register_VA(self, new_VA):
        self.formula.certificate.add(tuple(new_VA))
        if self.mode == "train":
            self.sat_history[self.sample_idx] = len(self.formula.certificate) - 1

    def get_dataset_stats(self):
        # Ratio of resolved SAT formulas
        sat_stats = {}
        re_count = [s for s in self.sat_history.values() if s > 0]
        resolved = len(re_count)
        total_sat = len([p for p in self.dataset["train"]["problems"] if p.is_sat])
        if total_sat > 0:
            sat_stats["resolved"] = resolved/total_sat
            sat_stats["VAs_per_formula"] = np.mean(re_count)
        return sat_stats

    
    def replace_proof(self, new_proof):
        old_proof = self.dataset[self.mode]["problems"][self.sample_idx].certificate
        self.dataset[self.mode]["problems"][self.sample_idx].certificate = new_proof
        self.reproven += self.sample_idx in self.stats[self.mode]
        shrink_trail = self.stats[self.mode].get(self.sample_idx, [len(old_proof)])
        self.stats[self.mode][self.sample_idx] = shrink_trail + [len(new_proof)]
        self.shrunk_res_size -= (len(old_proof) - len(new_proof))
        

    def resolve(self, c1, c2):
        c2_inv = set([-x for x in c2])
        common_lits = list(set(c1) & c2_inv)
        # No useful clause can be derived
        if len(common_lits) != 1: return None
        # Generate resultant clause
        new_clause = (set(c1) | set(c2)) - {common_lits[0], -common_lits[0]}
        new_clause = tuple(sorted(list(new_clause)))
        if new_clause in self.clause_map:
                return None
        return new_clause
    
    def build_clause_mat(self):
        '''
            Returns a matrix of clauses, where each row is a clause
            and each column is a variable (+1,-1)
        '''
        mat = np.zeros((len(self.clauses), self.formula.n_vars), dtype=int)
        for i, c in enumerate(self.clauses):
            mat[i] = self.clause_vec(c)
        self.clause_mat = mat
    
    def build_res_mask(self):
        '''
            Returns a mask of resolvable clause pairs
        '''
        F = self.clause_mat
        F_res = (F + F[:, None, :]).clip(-1, 1)
        res_mask = np.abs(F_res*F[None]).sum(axis=-1) == (np.abs(F).sum(axis=-1) - 1)

        self.F_res = F_res
        self.res_mask = res_mask
    
    # @profile
    def update_res_mask(self, c_new):
        '''
            Updates the res_mask after a new clause is added
        '''
        F = self.clause_mat
        c_new = self.clause_vec(c_new)
        F_res = (F + c_new[None]).clip(-1, 1).astype(int)
        res_mask = np.abs(F_res*F[None]).sum(axis=-1) == (np.abs(F).sum(axis=-1) - 1)
        res_mask = res_mask[0]

        self.res_mask = np.vstack((self.res_mask, res_mask[:-1]))
        self.res_mask = np.hstack((self.res_mask, res_mask[None].T))

    def clause_vec(self, c):
        '''
            Returns a vector of a single clause
        '''
        # c = np.asarray(c)
        vec = np.zeros(self.formula.n_vars)
        if len(c) == 0: return vec
        vec[np.abs(c) - 1] = np.sign(c)
        return vec
    
    def sep_clause_mat(self, c):
        '''
            Returns a vector of a list of clause
        '''
        return np.vstack([self.clause_vec(x) for x in c])
        
    def add_clause(self, c_new):
        self.clauses.append(c_new)
        self.clause_map[c_new] = len(self.clauses) - 1
        self.clause_mat = np.vstack((self.clause_mat, self.clause_vec(c_new)))
        self.update_res_mask(c_new)


    # @profile
    def add_clauses(self, c_new):
        if not isinstance(c_new, list):
            c_new = [c_new]
        # self.clause_map.update(c_new)
        self.clause_map.update({c: i + len(self.clauses) - 1 for i, c in enumerate(c_new)})
        self.clauses.extend(c_new)
        for c in c_new:
            self.clause_mat = np.vstack((self.clause_mat, self.clause_vec(c)))
            self.update_res_mask(c)
        # self.update_res_mask(c_new_mat)

    def format_proof(self):
        formatted_steps = []
        for step in self.res_trail:
            if step[2] is not None: result = f"({step[1]}): {step[2]}"
            else: result = "X"
            step_str = f"({step[0][0]}) & ({step[0][1]}) => {result}"
            formatted_steps.append(step_str)
        return formatted_steps
    
    def get_proof(self):
        res_in = lambda x: (self.clauses[x[0]], self.clauses[x[1]])
        proof = [(*res_in(step[0]), step[2]) for step in self.res_trail]
        return proof
    
    def num_resolvable_pairs(self):
        # Return the number of resolvable clause pairs
        cnt = 0
        for c1 in self.clauses:
            for c2 in self.clauses:
                if c1 == c2: continue
                if self.resolve(c1, c2) is not None:
                    cnt += 1
        return cnt/2
    
    def build_lookahead_mat(self):
        # F => (num_clauses, num_literals)
        F = self.clause_mat
        res_mask = self.res_mask
        F_res = self.F_res
        N, L = self.clause_mat.shape

        R_len = np.abs(F_res).sum(axis=-1)
        R_len[~res_mask] = L+1
        S1 = np.apply_along_axis(lambda x: np.bincount(x, minlength=L+2), axis=1, arr=R_len)
        S1 = S1[:, :-1]
        self.S1 = S1

    def update_lookahead_mat(self):
        # F => (num_clauses, num_literals)
        F = self.clause_mat
        F_res = self.F_res[-1]
        res_mask = self.res_mask[-1][:-1]
        S1 = self.S1
        N, L = F.shape
        
        R_len = np.abs(F_res).sum(axis=-1)[:-1]
        R_len[~res_mask] = L+1
        S = np.apply_along_axis(lambda x: np.bincount(x, minlength=L+2), axis=1, arr=R_len[:, None])
        S = S[:, :-1]
        C_score = S.sum(axis=0)
        C_score[len(self.clauses[-1])] += 1
        S1[res_mask, R_len[res_mask]] += 1
        S1_new = np.vstack([S1, C_score[None]])
        self.S1 = S1_new

    def which_res_var(self, idx1, idx2):
        """Determines which variable is being resolved on"""
        c1, c2 = self.clauses[idx1], self.clauses[idx2]
        c2_inv = set([-x for x in c2])
        common_lits = list(set(c1) & c2_inv)
        return abs(common_lits[0]) - 1
    
    def pivot_by_var(self, v):
        '''
            Returns the grid of clauses containing the pivot variable
            where the rows are the +ve literals and the columns are the -ve literals
        '''
        F = self.clause_mat

        v_pos_idx = np.where(F[:, v] == 1)[0]
        v_neg_idx = np.where(F[:, v] == -1)[0]
        v_res_mask = self.res_mask[v_pos_idx][:, v_neg_idx]

        ret_dict = {
            "v_pos_idx": v_pos_idx,
            "v_neg_idx": v_neg_idx,
            "v_res_mask": v_res_mask,
        }
        return ret_dict

    # =====================================================================
    # Batched API — operates on per-row state (`*_b` attributes) independent
    # of the single-formula instance attributes used by the legacy methods.
    # See plan: /Users/oskar/.claude/plans/i-want-to-batch-sequential-reddy.md
    # =====================================================================

    def _clause_vec_n(self, c, n_vars):
        vec = np.zeros(n_vars)
        if len(c) == 0:
            return vec
        vec[np.abs(c) - 1] = np.sign(c)
        return vec

    def _row_horizon(self, formula: Formula) -> int:
        if formula.is_sat:
            optim_factor = 2 if self.mode == "train" else 4
            H = formula.n_vars * optim_factor
            if self.config['max_H'] is not None:
                H = self.config['max_H']
        else:
            optim_factor = 3 if self.mode == "train" else 4
            if self.config['max_pLen'] is not None:
                optim_factor = self.config['max_pLen']
            H = len(formula.certificate) * optim_factor
        return H

    def _build_clause_mat_n(self, clauses, n_vars):
        mat = np.zeros((len(clauses), n_vars), dtype=int)
        for i, c in enumerate(clauses):
            mat[i] = self._clause_vec_n(c, n_vars)
        return mat

    def _build_res_mask(self, mat):
        F_res = (mat + mat[:, None, :]).clip(-1, 1)
        res_mask = np.abs(F_res * mat[None]).sum(axis=-1) == (np.abs(mat).sum(axis=-1) - 1)
        return F_res, res_mask

    def _make_guide_for_row(self, is_sat, solution, clauses_ref, clause_map_ref):
        # Closures capture per-row references; since clauses_ref / clause_map_ref
        # are the SAME mutable list/dict that add_clause_row mutates, the guide
        # sees up-to-date clause indices as new resolvents are added.
        if self.supervised:
            def gen():
                if is_sat:
                    while True:
                        yield {"VA": solution, "Res": None}
                else:
                    derived_map = {s[1]: s[2] for s in solution}
                    cform = lambda c: derived_map[c] if c in derived_map else clauses_ref[c]
                    for step in solution:
                        c1, c2 = cform(step[0][0]), cform(step[0][1])
                        c_idx = (clause_map_ref[c1], clause_map_ref[c2])
                        yield {"VA": None, "Res": c_idx}
            return gen()
        else:
            def gen():
                while True:
                    yield {"VA": None, "Res": None}
            return gen()

    def init_batch(self, formulas: List[Formula], sample_idxs: Optional[List[int]] = None) -> BatchedFormula:
        """Initialize per-row state for a batch and return the padded snapshot.
        sample_idxs is optional and only needed for replace_proof_at."""
        B = len(formulas)
        self.batch_size_b = B
        self.formulas_b = list(formulas)
        self.sample_idx_b = list(sample_idxs) if sample_idxs is not None else [None] * B
        self.is_sat_b = [f.is_sat for f in formulas]
        self.solution_b = [f.certificate for f in formulas]

        self.clauses_b = [list(f.clauses) for f in formulas]
        self.clause_map_b = [{c: i for i, c in enumerate(cs)} for cs in self.clauses_b]
        self.clause_mat_b = []
        self.F_res_b = []
        self.res_mask_b = []
        for b, f in enumerate(formulas):
            mat = self._build_clause_mat_n(self.clauses_b[b], f.n_vars)
            F_res, rm = self._build_res_mask(mat)
            self.clause_mat_b.append(mat)
            self.F_res_b.append(F_res)
            self.res_mask_b.append(rm)

        self.H_b = [self._row_horizon(f) for f in formulas]
        self.timeout_count_b = list(self.H_b)
        self.res_trail_b = [[] for _ in range(B)]
        self.found_sat_VA_b = [False] * B
        self.sat_VA_parity_b = [-1] * B
        self.guide_b = [
            self._make_guide_for_row(self.is_sat_b[b], self.solution_b[b],
                                     self.clauses_b[b], self.clause_map_b[b])
            for b in range(B)
        ]
        self.done_b = np.zeros(B, dtype=bool)
        self.solved_b = np.zeros(B, dtype=bool)
        self.ep_len_b = np.zeros(B, dtype=int)

        # Build the padded snapshot. H_max = max_b(initial n_clauses + horizon).
        H_max = max(len(self.clauses_b[b]) + self.H_b[b] for b in range(B))
        device = th.device(self.config['device'])
        self.bf = batch_formulas(formulas, device, H_max)
        self.max_vars_b = self.bf.max_vars
        return self.bf

    def all_done(self) -> bool:
        return bool(self.done_b.all())

    def running_mask_t(self) -> th.BoolTensor:
        device = th.device(self.config['device'])
        return th.from_numpy(~self.done_b).to(device)

    def guide_step_batch(self) -> List[dict]:
        out = []
        for b in range(self.batch_size_b):
            if self.done_b[b]:
                out.append({"VA": None, "Res": None})
                continue
            try:
                out.append(next(self.guide_b[b]))
            except StopIteration:
                # Guide exhausted before solver finished — supervised UNSAT proof
                # ran out of steps. Fall back to no-op so the batch can finish.
                out.append({"VA": None, "Res": None})
        return out

    def _add_clause_row(self, b: int, c_new: tuple) -> int:
        """Append c_new to row b's clause set, updating clause_map / clause_mat /
        res_mask. Returns the new clause's index."""
        f = self.formulas_b[b]
        idx = len(self.clauses_b[b])
        self.clauses_b[b].append(c_new)
        self.clause_map_b[b][c_new] = idx
        new_vec = self._clause_vec_n(c_new, f.n_vars)
        self.clause_mat_b[b] = np.vstack((self.clause_mat_b[b], new_vec))
        # Incremental res_mask update (mirrors update_res_mask logic)
        F = self.clause_mat_b[b]
        F_res_row = (F + new_vec[None]).clip(-1, 1).astype(int)
        rm_row = np.abs(F_res_row * F[None]).sum(axis=-1) == (np.abs(F).sum(axis=-1) - 1)
        rm_row = rm_row[0]
        self.res_mask_b[b] = np.vstack((self.res_mask_b[b], rm_row[:-1]))
        self.res_mask_b[b] = np.hstack((self.res_mask_b[b], rm_row[None].T))
        return idx

    def _resolve_row(self, b: int, c1: tuple, c2: tuple) -> Optional[tuple]:
        c2_inv = set([-x for x in c2])
        common_lits = list(set(c1) & c2_inv)
        if len(common_lits) != 1:
            return None
        new_clause = (set(c1) | set(c2)) - {common_lits[0], -common_lits[0]}
        new_clause = tuple(sorted(list(new_clause)))
        if new_clause in self.clause_map_b[b]:
            return None
        return new_clause

    def _is_satisfying_row(self, b: int, VAs) -> bool:
        if VAs is None:
            return False
        f = self.formulas_b[b]
        N = f.n_clauses
        orig_mat = self.clause_mat_b[b][:N]
        for i, VA in enumerate(VAs):
            if isinstance(VA, th.Tensor):
                VA = VA.squeeze(0).detach().cpu().numpy()
            VA_sign = VA.copy()
            VA_sign[VA_sign == 0] = -1
            sub_mat = orig_mat * VA_sign
            satisfied = np.any(sub_mat == 1, axis=-1)
            if np.all(satisfied):
                self.sat_VA_parity_b[b] = i
                return True
        return False

    def step_batch(self, actions: List[dict]) -> Tuple[List, np.ndarray, np.ndarray, List[dict]]:
        """Per-row step. actions[b] = {"Res": (i,j) or list-of-pairs or None,
        "VA": list-of-VAs or None}. Skips done rows. Returns
        (new_clauses_per_row, rewards, done_mask, infos_per_row)."""
        B = self.batch_size_b
        assert len(actions) == B, f"Expected {B} actions, got {len(actions)}"
        new_clauses_per_row: List[List[tuple]] = [[] for _ in range(B)]
        new_indices_per_row: List[List[int]] = [[] for _ in range(B)]
        rewards = np.zeros(B, dtype=float)
        infos: List[dict] = [{} for _ in range(B)]

        for b in range(B):
            if self.done_b[b]:
                infos[b] = {"solved": bool(self.solved_b[b])}
                continue
            self.timeout_count_b[b] -= 1
            self.ep_len_b[b] += 1
            act = actions[b]
            Res = act.get("Res")
            VAs = act.get("VA")

            new_cs: List[tuple] = []
            if Res is not None:
                # Normalize to iterable of pairs.
                pairs = Res if isinstance(Res, list) else [Res]
                for r in pairs:
                    if r is None:
                        continue
                    i, j = r
                    if i < 0 or j < 0:
                        continue
                    c1 = self.clauses_b[b][i]
                    c2 = self.clauses_b[b][j]
                    if 0 <= i < len(self.res_mask_b[b]) and 0 <= j < len(self.res_mask_b[b]):
                        self.res_mask_b[b][i, j] = False
                        self.res_mask_b[b][j, i] = False
                    new_clause = self._resolve_row(b, c1, c2)
                    if new_clause is not None:
                        new_idx = self._add_clause_row(b, new_clause)
                        self.res_trail_b[b].append((r, new_idx, new_clause))
                        new_cs.append(new_clause)
                        new_indices_per_row[b].append(new_idx)
            new_clauses_per_row[b] = new_cs

            proven_unsat = any(c is not None and len(c) == 0 for c in new_cs)
            if self.found_sat_VA_b[b]:
                proven_sat = True
            else:
                proven_sat = self.is_sat_b[b] and self._is_satisfying_row(b, VAs)
            solved = proven_unsat or proven_sat
            timeout = self.timeout_count_b[b] <= 0
            done = solved or timeout
            self.solved_b[b] = bool(solved)
            self.done_b[b] = bool(done)
            infos[b] = {"solved": bool(solved)}

        # Caller (NeuRes / MP_Embedder) needs to know which row indices got new
        # clauses so it can update its padded clause_emb / L_unpack columns.
        return new_clauses_per_row, rewards, self.done_b.copy(), infos

    def replace_proof_at(self, b: int, new_proof):
        """Per-row variant of replace_proof — mutates dataset for row b."""
        sample_idx = self.sample_idx_b[b]
        if sample_idx is None:
            return
        problems = self.dataset[self.mode]["problems"]
        old_proof = problems[sample_idx].certificate
        problems[sample_idx].certificate = new_proof
        self.reproven += sample_idx in self.stats[self.mode]
        shrink_trail = self.stats[self.mode].get(sample_idx, [len(old_proof)])
        self.stats[self.mode][sample_idx] = shrink_trail + [len(new_proof)]
        self.shrunk_res_size -= (len(old_proof) - len(new_proof))

    def pivot_by_var_batched(self, var_idx_t: th.LongTensor) -> dict:
        """For AnchAttention. Per row, pick clauses containing literal var_idx[b]
        positively (pos_idx) and negatively (neg_idx); pad to (B, max_pos) /
        (B, max_neg). v_res_mask[b, p, n] = True iff (pos_idx[b,p], neg_idx[b,n])
        is currently resolvable."""
        B = self.batch_size_b
        var_idx = var_idx_t.detach().cpu().numpy()
        pos_lists, neg_lists, res_masks = [], [], []
        for b in range(B):
            v = int(var_idx[b])
            F = self.clause_mat_b[b]
            if v >= F.shape[1]:
                # Padded variable selected for this row — empty pivot.
                pos_lists.append(np.empty(0, dtype=int))
                neg_lists.append(np.empty(0, dtype=int))
                res_masks.append(np.zeros((0, 0), dtype=bool))
                continue
            pos = np.where(F[:, v] == 1)[0]
            neg = np.where(F[:, v] == -1)[0]
            pos_lists.append(pos)
            neg_lists.append(neg)
            if len(pos) > 0 and len(neg) > 0:
                rm = self.res_mask_b[b][pos][:, neg]
            else:
                rm = np.zeros((len(pos), len(neg)), dtype=bool)
            res_masks.append(rm)

        max_pos = max((len(p) for p in pos_lists), default=0)
        max_neg = max((len(n) for n in neg_lists), default=0)
        # Allocate at least 1 along each dim so downstream gathers don't choke
        # on empty tensors; the masks will hide all entries.
        Pp = max(max_pos, 1)
        Nn = max(max_neg, 1)
        device = th.device(self.config['device'])
        pos_idx = th.zeros(B, Pp, dtype=th.long, device=device)
        neg_idx = th.zeros(B, Nn, dtype=th.long, device=device)
        pos_mask = th.zeros(B, Pp, dtype=th.bool, device=device)
        neg_mask = th.zeros(B, Nn, dtype=th.bool, device=device)
        v_res_mask = th.zeros(B, Pp, Nn, dtype=th.bool, device=device)
        for b in range(B):
            pl, nl = len(pos_lists[b]), len(neg_lists[b])
            if pl > 0:
                pos_idx[b, :pl] = th.from_numpy(pos_lists[b]).to(device)
                pos_mask[b, :pl] = True
            if nl > 0:
                neg_idx[b, :nl] = th.from_numpy(neg_lists[b]).to(device)
                neg_mask[b, :nl] = True
            if pl > 0 and nl > 0:
                v_res_mask[b, :pl, :nl] = th.from_numpy(res_masks[b]).to(device)
        return {
            "pos_idx": pos_idx, "neg_idx": neg_idx,
            "pos_mask": pos_mask, "neg_mask": neg_mask,
            "v_res_mask": v_res_mask,
            "max_pos": Pp, "max_neg": Nn,
        }

    def which_res_var_batched(self, idx1_t: th.LongTensor, idx2_t: th.LongTensor) -> th.LongTensor:
        """Per row, return the pivot variable for resolving (idx1[b], idx2[b])."""
        B = self.batch_size_b
        i1 = idx1_t.detach().cpu().numpy()
        i2 = idx2_t.detach().cpu().numpy()
        out = np.zeros(B, dtype=np.int64)
        for b in range(B):
            if self.done_b[b]:
                continue
            try:
                c1 = self.clauses_b[b][int(i1[b])]
                c2 = self.clauses_b[b][int(i2[b])]
                c2_inv = set([-x for x in c2])
                common = list(set(c1) & c2_inv)
                if len(common) == 1:
                    out[b] = abs(common[0]) - 1
            except IndexError:
                pass
        device = th.device(self.config['device'])
        return th.from_numpy(out).to(device)

    def res_mask_batched_t(self, current_C: int) -> th.BoolTensor:
        """Stack per-row res_mask into a (B, current_C, current_C) bool tensor,
        padded with False. current_C must be >= max_b(len(res_mask_b[b]))."""
        B = self.batch_size_b
        device = th.device(self.config['device'])
        out_np = np.zeros((B, current_C, current_C), dtype=bool)
        for b in range(B):
            n = self.res_mask_b[b].shape[0]
            if n > 0:
                out_np[b, :n, :n] = self.res_mask_b[b]
        return th.from_numpy(out_np).to(device)

    def iter_batches(self, batch_size: int):
        """Generator yielding (formulas, sample_idxs) pairs of size up to
        batch_size from the current mode's chunked sample_idxs."""
        idxs = list(range(self.st_idx, self.end_idx))
        if self.shuffle:
            random.shuffle(idxs)
        problems = self.dataset[self.mode]["problems"]
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start:start + batch_size]
            formulas = [problems[i] for i in batch_idxs]
            yield formulas, batch_idxs