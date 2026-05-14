from solvers import solver_map
from envs.res_unsat import ResUNSAT
from common.utils import RolloutData, compute_returns, safe_mean
import torch as th
import numpy as np
from tqdm import tqdm
import os
from common.problem import Formula, reduce_res_tree, remove_idx_gaps
from common.logger import Logger
from common.buffer import RolloutBuffer
from common.utils import Episode
from solvers import BaseSolver
import pickle as pkl


class TeacherForce():
    def __init__(self, config, env: ResUNSAT) -> None:
        self.config = config
        self.solver: BaseSolver = solver_map[config['solver']](config, env)
        self.env = env
        self.exp_name = config['exp_name']
        self.no_valid = config['no_valid']
        
        self.gamma = self.solver.gamma
        self.val_freq = config['val_freq']
        if not config['test_only']:
            self.rollout_buffer = RolloutBuffer(config)
        else:
            self.rollout_buffer = None
        self.best_success_rate = 0.0

        self.model_name = f"{config['exp_name']}_{config['variant']}"


    def _setup_buffers(self):
        self.rollout_buffer.reset()
    
    def _add_rollout_stat(self, log_dict, mode, tag, roll_window=20):
        ep_count = self.solver.ep_count
        self.rollout_buffer.add_rollout_stat(log_dict, mode, tag, ep_count, roll_window=roll_window)
        
    @th.no_grad()
    def preroll_shrink_batch(self, formulas, sample_idxs):
        """Run B formulas unsupervised in lockstep, then per-row replace any
        formula's certificate with a shorter proof if the model found one.
        SAT rows are marked done up-front so they sit out the preroll. Returns
        a (possibly skip-filtered) list of (formula, sample_idx) to actually
        train on. shrink_skip (per the original semantics) drops shrunk rows
        from the train batch."""
        self.solver.reset(train=False)
        self.solver.eval()
        self.env.save_iter_state()
        self.env.supervised = False

        self.solver.setup_batch(formulas, sample_idxs=sample_idxs)
        # Tighter timeout: stop the model as soon as it has spent the same
        # number of steps as the recorded proof length minus one.
        B = len(formulas)
        for b, f in enumerate(formulas):
            if f.is_sat:
                self.env.done_b[b] = True
            else:
                self.env.timeout_count_b[b] = max(0, len(f.certificate) - 1)

        while not self.env.all_done():
            guide_acts = self.env.guide_step_batch()
            actions = self.solver.resolve_batch(guide_acts)
            self.env.step_batch(actions)

        # Per-row decide whether to replace the dataset's certificate.
        keep_for_train = [True] * B
        any_shrunk = False
        for b, f in enumerate(formulas):
            if f.is_sat:
                continue
            if not bool(self.env.solved_b[b]):
                continue
            new_proof = reduce_res_tree(self.env.res_trail_b[b])
            new_proof = remove_idx_gaps(new_proof, len(f.clauses))
            if len(new_proof) < len(f.certificate):
                self.env.replace_proof_at(b, new_proof)
                any_shrunk = True
                if self.config['shrink_skip']:
                    keep_for_train[b] = False
        if any_shrunk:
            shrink_factor = self.env.shrunk_res_size / max(self.env.orig_res_size, 1)
            reproven_ratio = self.env.reproven / max(self.env.total_problems, 1)
            self._add_rollout_stat(
                {"shrink_factor": shrink_factor, "re-proven": reproven_ratio},
                "train", "", roll_window=1,
            )

        self.env.restore_iter_state()
        self.solver.train()
        kept = [(formulas[b], sample_idxs[b]) for b in range(B) if keep_for_train[b]]
        return kept

    # @profile
    def train(self, n_epochs_plan: int, n_epochs_stop: int, tag: str = "", callback=None):

        self._setup_buffers()
        self.solver.train()
        self.env.set_mode("train")
        self.env.supervised = True
        bs = max(1, self.config['batch_size'])
        # ep_count semantics for the LR scheduler: number of optimizer steps
        # taken. Each batch is one accumulation slot; an optimizer step happens
        # every `update_every` batches.
        n_batches = max(1, len(self.env) // bs)
        self.solver.total_episodes = max(
            1, (n_batches * n_epochs_plan) // self.config['update_every']
        )
        self.solver.ep_count = 0

        periodic_val = self.val_freq is not None
        for epoch in range(n_epochs_stop):
            print(f"Epoch {epoch} started..")
            i = 0
            for formulas, sample_idxs in tqdm(self.env.iter_batches(bs), total=n_batches):
                # Optional expert-iteration preroll: drop shrunk-skip rows.
                if self.config['bootstrap_shrink'] and any(not f.is_sat for f in formulas):
                    kept = self.preroll_shrink_batch(formulas, sample_idxs)
                    if not kept:
                        i += 1
                        continue
                    formulas = [k[0] for k in kept]
                    sample_idxs = [k[1] for k in kept]

                ret_dict = self.run_batch(formulas, sample_idxs, train=True)
                # Sanity: supervised UNSAT rows should always solve.
                for b, f in enumerate(formulas):
                    if not f.is_sat:
                        assert ret_dict["solved"][b], \
                            f"Supervised UNSAT should always solve the formula (row {b})"

                if (i + 1) % self.config['update_every'] == 0:
                    loss_dict = self.solver.update_batch()
                    log_dict = {
                        'LR': self.solver.optimizer.param_groups[0]['lr'],
                        "loss": loss_dict,
                    }
                    if "pred_accr" in ret_dict:
                        log_dict["pred_accr"] = ret_dict["pred_accr"]
                    self._add_rollout_stat(log_dict, "train", tag, roll_window=20)
                    if periodic_val and i > 0 and i % self.val_freq == 0:
                        avg_solved = self.validate("val", tag)
                        if callback is not None:
                            callback(i, avg_solved)
                i += 1
            self.env.save_data_stats(model_name=self.model_name)
            # Flush any remaining accumulated batches.
            if len(self.solver.batch_data) > 0:
                _ = self.solver.update_batch()
            print(f"Epoch {epoch} ended!")
            if not self.no_valid and not periodic_val:
                avg_solved = self.validate("val", tag, save_model=True)
                if callback is not None:
                    callback(epoch, avg_solved)

    def parse_test_stats(self, stats):
        sat_pool = [x for x in stats if x["formula"].is_sat]
        unsat_pool = [x for x in stats if not x["formula"].is_sat]
        SR = {
            "SAT": [x["solved"] for x in sat_pool],
            "UNSAT": [x["solved"] for x in unsat_pool],
            "Total": [x["solved"] for x in stats],
        }
        if "pred_accr" in stats[0]:
            ACCR = {
                "SAT": [x["pred_accr"] for x in sat_pool],
                "UNSAT": [x["pred_accr"] for x in unsat_pool],
                "Total": [x["pred_accr"] for x in stats],
            }
        else:
            ACCR = {}
        OPT = {
            "SAT/a-Len": [x["ep_len"]/x["formula"].n_lits for x in sat_pool if x["solved"]],
            "UNSAT/ep-Len": [x["ep_len"]/len(x["formula"].certificate) for x in unsat_pool if x["solved"]],
            "UNSAT/p-Len": [x["proof_len"]/len(x["formula"].certificate) for x in unsat_pool if x["solved"]],
            "UNSAT/over-expert": [x["proof_len"]/len(x["formula"].certificate) < 1.0 for x in unsat_pool if x["solved"]],
            "UNSAT/over-expert-red": [x["proof_len"]/len(x["formula"].certificate) for x in unsat_pool if x["ep_len"]/len(x["formula"].certificate) < 1.0 and x["solved"]]
        }
        SIZE = {
            # Solved
            "SAT/solved/vars": [x["formula"].n_vars for x in sat_pool if x["solved"]],
            "SAT/solved/clauses": [x["formula"].n_clauses for x in sat_pool if x["solved"]],
            "UNSAT/solved/vars": [x["formula"].n_vars for x in unsat_pool if x["solved"]],
            "UNSAT/solved/clauses": [x["formula"].n_clauses for x in unsat_pool if x["solved"]],
            # Unsolved
            "SAT/unsolved/vars": [x["formula"].n_vars for x in sat_pool if not x["solved"]],
            "SAT/unsolved/clauses": [x["formula"].n_clauses for x in sat_pool if not x["solved"]],
            "UNSAT/unsolved/vars": [x["formula"].n_vars for x in unsat_pool if not x["solved"]],
            "UNSAT/unsolved/clauses": [x["formula"].n_clauses for x in unsat_pool if not x["solved"]],
        }
        TIME = {
            # Add time stats
        }
        log_dict_pre = {"SR": SR, "ACCR": ACCR, "OPT": OPT, "SIZE": SIZE, "TIME": TIME}
        log_dict = {}
        print("Test stats:")
        for k in log_dict_pre:
            for k_ in log_dict_pre[k]:
                if len(log_dict_pre[k][k_]) > 0 or "UNSAT" in k_:
                    log_dict[f"{k}/{k_}"] = safe_mean(log_dict_pre[k][k_])
                    print(f"{k}/{k_}: {log_dict[f'{k}/{k_}']}")
        
        return log_dict
    
    @th.no_grad()
    def validate(self, mode, tag: str = "", save_model=True):
        print(f"Validating on {mode} set..")
        self.solver.eval()
        self.env.save_iter_state()
        self.env.set_mode(mode)
        self.env.supervised = False
        bs = max(1, self.config['batch_size'])
        n_batches = max(1, len(self.env) // bs)
        stats = []
        for formulas, sample_idxs in tqdm(self.env.iter_batches(bs), total=n_batches):
            ret_dict = self.run_batch(formulas, sample_idxs, train=False)
            for b, f in enumerate(formulas):
                stats.append({
                    "formula": f,
                    "solved": ret_dict["solved"][b],
                    "ep_len": ret_dict["ep_len"][b],
                    "proof_len": ret_dict["proof_len"][b],
                })

        log_dict = self.parse_test_stats(stats)
        print(f"Validation on {mode} set ended!")
        self._add_rollout_stat(log_dict, f"valid_{mode}", tag, roll_window=1)
        self.save_checkpoint(f"latest_model")
        if save_model:
            if log_dict["SR/Total"] > self.best_success_rate:
                self.best_success_rate = log_dict["SR/Total"]
                self.save_checkpoint("best_model")
            if self.config['save_all_checkpoints']:
                self.save_checkpoint(f"ep_{self.solver.ep_count}")
        self.solver.train()
        self.env.restore_iter_state()
        return log_dict["SR/Total"]

    @th.no_grad()
    def test(self, chunk_id=''):
        print(f"Testing on {self.env.variant} starts..")
        self.solver.eval()
        self.env.save_iter_state()
        self.env.set_mode("test")
        self.env.supervised = False
        bs = max(1, self.config['batch_size'])
        n_batches = max(1, len(self.env) // bs)
        res = []
        for formulas, sample_idxs in tqdm(self.env.iter_batches(bs), total=n_batches):
            try:
                ret_dict = self.run_batch(formulas, sample_idxs, train=False)
                for b, f in enumerate(formulas):
                    extras = {"sat_VA_parity": self.env.sat_VA_parity_b[b]}
                    episode = Episode(
                        sample_idxs[b], f,
                        ret_dict["solved"][b],
                        ret_dict["ep_len"][b],
                        extras,
                    )
                    res.append(episode)
            except Exception as e:
                print(f"Error on batch (rows {sample_idxs}): {e}")
                for b, f in enumerate(formulas):
                    res.append(Episode(sample_idxs[b], f, False, np.inf, {}))
            if (len(res) + 1) % 300 == 0:
                print(f"Saving eval backup at {len(res)} steps..")
                os.makedirs(f"{self.config['eval_dir']}/{self.config['dataset']}/", exist_ok=True)
                pkl.dump(res, open(
                    f"{self.config['eval_dir']}/{self.config['dataset']}/{self.eval_model_name}{chunk_id}.pkl",
                    "wb",
                ))
        print(f"Testing on {self.env.variant} ended!")
        self.env.restore_iter_state()
        return res

    def save_checkpoint(self, tag):
        os.makedirs(f"checkpoints/{self.config['exp_name']}/{self.config['variant']}", exist_ok=True)
        mpath = f"checkpoints/{self.config['exp_name']}/{self.config['variant']}/{tag}.pth"
        save_dict = {
            "solver": self.solver.state_dict(),
            "optimizer": self.solver.optimizer.state_dict(),
        }
        th.save(save_dict, mpath)


    def run_batch(self, formulas, sample_idxs, train) -> dict:
        """Run B formulas in lockstep until env.all_done(); finalize per-batch
        loss buffers; return per-row stats."""
        self.solver.reset(train)
        self.solver.setup_batch(formulas, sample_idxs=sample_idxs)
        while not self.env.all_done():
            guide_acts = self.env.guide_step_batch()
            actions = self.solver.resolve_batch(guide_acts)
            self.env.step_batch(actions)
        self.solver.finalize_batch()

        B = len(formulas)
        proof_lens = [
            len(self.env.res_trail_b[b]) if bool(self.env.solved_b[b]) else 0
            for b in range(B)
        ]
        return {
            "solved": [bool(s) for s in self.env.solved_b],
            "ep_len": [int(x) for x in self.env.ep_len_b],
            "proof_len": proof_lens,
        }
    
