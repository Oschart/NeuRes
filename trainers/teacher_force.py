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
    def __init__(self, opt, env: ResUNSAT) -> None:
        self.opt = opt
        self.solver: BaseSolver = solver_map[opt.solver](opt, env)
        self.env = env
        self.exp_name = opt.exp_name
        self.no_valid = opt.no_valid
        
        self.gamma = self.solver.gamma
        self.val_freq = opt.val_freq
        if not opt.test_only:
            self.rollout_buffer = RolloutBuffer(opt)
        else:
            self.rollout_buffer = None
        self.best_success_rate = 0.0

        self.model_name = f"{opt.exp_name}_{opt.variant}"


    def _setup_buffers(self):
        self.rollout_buffer.reset()
    
    def _add_rollout_stat(self, log_dict, mode, tag, roll_window=20):
        ep_count = self.solver.ep_count
        self.rollout_buffer.add_rollout_stat(log_dict, mode, tag, ep_count, roll_window=roll_window)
        
    @th.no_grad()
    def preroll_shrink(self, problem: Formula):
        self.solver.reset(train=False)
        self.solver.eval()
        self.env.save_iter_state()
        self.env.supervised = False

        self.solver.setup_formula(problem)
        self.env.timeout_count = len(self.env.solution) - 1
        done = False
        while not done:
            action = self.solver.resolve({"VA": None, "Res": None})
            _, _, done, info = self.env.step(action)
        # new_proof = [t[0] for t in self.env.res_trail]
        new_proof = self.env.res_trail
        new_proof = reduce_res_tree(new_proof)
        new_proof = remove_idx_gaps(new_proof, len(problem.clauses))
        if info["solved"] and len(new_proof) < len(self.env.solution):
            self.env.replace_proof(new_proof)
            shrink_factor = self.env.shrunk_res_size/self.env.orig_res_size
            reproven_ratio = self.env.reproven/self.env.total_problems
            self._add_rollout_stat({"shrink_factor": shrink_factor, "re-proven": reproven_ratio}, "train", "", roll_window=1)
            if self.opt.shrink_skip:
                self.env.restore_iter_state()
                self.solver.train()
                print(f"NeuRes proof was {shrink_factor:.2f}x shorter, skipping..")
                return None
        
        self.env.restore_iter_state()
        self.solver.train()
        problem = self.env.init_episode()
        return problem

    # @profile
    def train(self, n_epochs_plan: int, n_epochs_stop: int, tag: str = "supervised", callback=None):
        
        self._setup_buffers()
        self.solver.train()
        self.env.set_mode("train")
        self.env.supervised = True
        self.solver.total_episodes = len(self.env)*n_epochs_plan
        self.solver.ep_count = 0

        periodic_val = self.val_freq is not None
        for epoch in range(n_epochs_stop):
            print(f"Epoch {epoch} started..")
            for i, problem in tqdm(enumerate(self.env), total=len(self.env)):
                if not problem.is_sat and self.opt.bootstrap_shrink:
                    problem = self.preroll_shrink(problem)
                if problem is None:
                    continue

                ret_dict = self.run_episode(problem, train=True)
                solved = ret_dict["solved"]
                assert problem.is_sat or solved == 1, "Supervised UNSAT should always solve the formula"
                    
                if (i+1) % self.opt.update_every == 0:
                    loss_dict = self.solver.update()
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
            self.env.save_data_stats(model_name=self.model_name)
            # Flush out the buffer
            if len(self.solver.ep_data["ep_len"]) > 0:
                _ = self.solver.update()
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
        stats = []
        for episode in tqdm(self.env):
            ret_dict = self.run_episode(episode, train=False)
            solved = ret_dict["solved"]
            extra_steps = ret_dict["ep_len"]
            stats.append({"formula": episode, **ret_dict})

        log_dict = self.parse_test_stats(stats)
        print(f"Validation on {mode} set ended!")
        self._add_rollout_stat(log_dict, f"valid_{mode}", tag, roll_window=1)
        self.save_checkpoint(f"latest_model")
        if save_model:
            if log_dict["SR/Total"] > self.best_success_rate:
                self.best_success_rate = log_dict["SR/Total"]
                self.save_checkpoint("best_model")
            if self.opt.save_all_checkpoints:
                self.save_checkpoint(f"ep_{self.solver.ep_count}")
        # if self.opt.bootstrap_shrink:
        #     self.env.save_data_stats(model_name=self.model_name)
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
        res = []
        for formula in tqdm(self.env):
            try:
                ret_dict = self.run_episode(formula, train=False)
                solved = ret_dict["solved"]
                ep_len = ret_dict["ep_len"]
            except Exception as e:
                print(f"Error on {formula}: {e}")
                solved, ep_len = False, np.inf
            if self.opt.track_VAs and formula.is_sat:
                unique_ratio = len(self.env.VAs)/ep_len
                extras = {"unique_VAs": self.env.unique_VAs, "unique_ratio": unique_ratio, "sat_progress": self.env.sat_progress}
            else:
                extras = {}
            extras["sat_VA_parity"] = self.env.sat_VA_parity
            episode = Episode(self.env.sample_idx, formula, solved, ep_len, extras)
            res.append(episode)
            # res.append((self.env.sample_idx, formula.n_vars, formula.n_clauses, solved, ep_len, len(formula.certificate)))
            if (len(res)+1)%300 == 0:
                print(f"Saving eval backup at {len(res)} steps..")
                os.makedirs(f"{self.opt.eval_dir}/{self.opt.dataset}/", exist_ok=True)
                pkl.dump(res, open(f"{self.opt.eval_dir}/{self.opt.dataset}/{self.eval_model_name}{chunk_id}.pkl", "wb"))
        print(f"Testing on {self.env.variant} ended!")
        self.env.restore_iter_state()
        return res

    def save_checkpoint(self, tag):
        os.makedirs(f"checkpoints/{self.opt.exp_name}/{self.opt.variant}", exist_ok=True)
        mpath = f"checkpoints/{self.opt.exp_name}/{self.opt.variant}/{tag}.pth"
        save_dict = {
            "solver": self.solver.state_dict(),
            "optimizer": self.solver.optimizer.state_dict(),
        }
        th.save(save_dict, mpath)


    def run_episode(self, formula: Formula, train):
        rewards = []
        self.solver.reset(train)
        
        self.solver.setup_formula(formula)
        done = False
        while not done:
            expert_action = self.env.guide_step()
            if train and self.opt.topk_train and self.opt.topk > 1 and not formula.is_sat:
                # Check if expert action has already been taken
                resp = expert_action["Res"]
                if not self.env.res_mask[resp[0], resp[1]]:
                    # print(f"Expert action {expert_action} already taken!")
                    continue
            action = self.solver.resolve(expert_action)
            c_new, reward, done, info = self.env.step(action)
            rewards.append(reward)
        
        solved = info["solved"]
        if solved:
            new_proof = self.env.res_trail
            # Uncomment if you want to eliminate redundant steps from generated proof (complexity: O(n))
            # new_proof = reduce_res_tree(new_proof)
            proof_len = len(new_proof)
        else:
            proof_len = 0
        
        rewards = np.array(rewards)
        ep_len = len(rewards)
        self.solver.post_episode(rewards)
        ret_dict = {
            "rewards": rewards,
            "solved": solved,
            "ep_len": ep_len,
            "proof_len": proof_len,
        }
        if "sat_pred" in action:
            ret_dict["pred_accr"] = action["sat_pred"] == formula.is_sat
        return ret_dict
    
