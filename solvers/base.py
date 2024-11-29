import torch as th
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, NamedTuple, Union
from networks.neu_res import NeuRes
from envs import ResUNSAT
from common.utils import Scheduler, PolicySpec, RolloutData
from common.problem import Formula
from common.utils import get_time_horizon
from torch.nn.functional import binary_cross_entropy_with_logits


class BaseSolver(nn.Module):
    def __init__(
        self,
        opt: PolicySpec,
        env: ResUNSAT,
    ):
        super(BaseSolver, self).__init__()
        self.opt = opt
        self.env = env
        self.opt.env = env
        self.lr_schedule = opt.lr_schedule
        self.gamma = opt.gamma
        self.max_grad_norm = opt.max_grad_norm
        self.device = opt.device
        self.all_sat = "OmNeuRes" in opt.solver

        self.opt.critic = False
        self._setup_model()
        self.load_checkpoint(opt.checkpoint)
        self.apply_freezers(opt.freeze)
        self.setup_losses()
        self.total_episodes = 0
        self.ep_count = 0
        self.reset()
        self.to(opt.device)

    def load_checkpoint(self, chkpt_path):
        if chkpt_path is None:
            return
        try:
            load_dict = th.load(chkpt_path)
            if "optimizer" in load_dict:
                if "solver" in load_dict:
                    self.load_state_dict(load_dict["solver"], strict=False)
                else:
                    self.load_state_dict(load_dict["agent"], strict=False)
                try:
                    self.optimizer.load_state_dict(load_dict["optimizer"])
                    self.lr_schedule.v = self.optimizer.param_groups[0]['lr']
                    print(f"Loaded optimizer from checkpoint at: {chkpt_path}")
                except Exception as e:
                    print(f"Error loading optimizer from checkpoint at: {chkpt_path}: {e}")
                    print("Skipping optimizer load.")
                
                
            else:
                self.load_state_dict(load_dict, strict=False)
            print(f"Loaded model from checkpoint at: {chkpt_path}")
        except Exception as e:
            print(f"Error loading: {e}")
            print(f"No valid model found at {chkpt_path}")
        # if self.opt.finetune_schedule is not None:
        #     self.optimizer

    def apply_freezers(self, freeze_list):
        if len(freeze_list) == 0: return
        frozen_params = set()
        for name, param in self.named_parameters():
            for freeze_name in freeze_list:
                if freeze_name in name:
                    param.requires_grad = False
                    frozen_params.add(freeze_name)

        print(f"Frozen params: {frozen_params}")

        # Update optimizer
        self.optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr_schedule(1), eps=1e-5)
        self.lr_schedule.v = self.optimizer.param_groups[0]['lr']
    
    def update_lr(self):
        rem_ratio = (self.total_episodes - self.ep_count) / self.total_episodes
        rem_ratio = max(1e-8, rem_ratio)
        self.optimizer.param_groups[0]['lr'] = self.lr_schedule(rem_ratio)


    def setup_losses(self):
        self.loss_funcs = {
            "sat": {
                "sat_loss": self.get_sat_loss,
            },
            "unsat": {
                "res_loss": self.get_res_loss,
            }
        }

    
    def get_loss_data(self):
        loss_data = {}
        TH = []
        for n in self.ep_data["ep_len"]:
            TH.append(get_time_horizon(n, self.gamma))
        TH = th.cat(TH).float().to(self.device)
        grad_keys = ["c_logp", "sat_loss", "sat_vote"]
        for k in grad_keys:
            if len(self.ep_data[k]) == 0: continue
            loss_data[k] = th.cat(self.ep_data[k])
        loss_data["TH"] = TH
        return loss_data
    
    def get_res_loss(self, loss_data: dict):
        TH = loss_data["TH"]
        res_loss = -(TH * loss_data["c_logp"]).mean()
        return res_loss
    
    def get_sat_loss(self, loss_data: dict):
        TH = loss_data["TH"]
        sat_loss = (TH * loss_data["sat_loss"]).mean()
        return sat_loss


    def compute_loss(self, no_grad=False):
        loss_data = self.get_loss_data()
        loss_dict = {}
        total_loss = 0.0
        sat_status = "sat" if self.is_sat else "unsat"

        if self.opt.predict_sat:
            sat_vote = loss_data["sat_vote"].mean()
            sat_gt = th.ones_like(sat_vote) if self.is_sat else th.zeros_like(sat_vote)
            sat_pred_loss = binary_cross_entropy_with_logits(sat_vote, sat_gt)
            loss_dict["sat_pred"] = sat_pred_loss
            total_loss += sat_pred_loss

        for k, loss_func in self.loss_funcs[sat_status].items():
            loss_dict[k] = loss_func(loss_data)
            total_loss += loss_dict[k]
        loss_dict["total"] = total_loss

        if no_grad:
            for k, v in loss_dict.items():
                if not isinstance(v, float):
                    loss_dict[k] = v.item()
        return loss_dict

    # @profile
    def update(self):
        loss_dict = self.compute_loss()
        total_loss = loss_dict["total"]
        self.optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        if th.isnan(total_loss):
            print("="*50)
            print("NaN detected in loss. Skipping update.")
            print("="*50)
        else:
            self.optimizer.step()
        self.ep_count += 1
        self.update_lr()

        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
        self.reset_buffers()
        return loss_dict
    
    def reset_buffers(self):
        self.ep_data = {"c_logp": [], "c_idx": [], "sat_loss": [], "sat_vote": [], "ep_len": []}
    
    def reset(self, train: bool = True):
        self.reset_buffers()
        self.policy.train(train)
        self.policy.reset()
    
    def post_episode(self, rewards):
        self.ep_data["ep_len"].append(len(rewards))
        return

    def _setup_model(self):
        self.policy = NeuRes(self.opt)
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr_schedule(1), eps=1e-5)

    def setup_formula(self, formula: Formula):
        self.is_sat = formula.is_sat
        self.policy.setup_formula(formula)
    
    def resolve(self, guide_act):
        ret_dict = self.policy(guide_act)
        for k in ret_dict:
            if k in self.ep_data:
                if isinstance(ret_dict[k], list):
                    self.ep_data[k].extend(ret_dict[k])
                else:
                    self.ep_data[k].append(ret_dict[k])
        action = {}
        if "c_idx" in ret_dict:
            c_inds = ret_dict["c_idx"]
            self.policy.add_new_clause(c_inds)
            action["Res"] = c_inds
        else:
            # Perform one message-passing round
            self.policy.perform_mp_round()
        if "A_pred" in ret_dict:
            action["VA"] = ret_dict["A_pred"]

        if self.opt.predict_sat:
            sat_pred = th.sigmoid(ret_dict["sat_vote"])
            action["sat_pred"] = int(th.round(sat_pred).item())

        return action
    
    def forward(self, guide_act=None):
        guide_act={"Res": None, "VA": None}
        return self.resolve(guide_act)
