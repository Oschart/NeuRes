import contextlib
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


_AUTOCAST_DTYPE = {"bf16": th.bfloat16, "fp16": th.float16}


def _make_autocast_ctx(device, autocast_dtype: str):
    """Context manager that enables autocast on CUDA when requested. CPU runs
    keep fp32 — bf16 on CPU needs PyTorch >= 2.0 with x86 support and offers
    little speedup for our model sizes."""
    if autocast_dtype == "none" or str(device) != "cuda":
        return contextlib.nullcontext()
    return th.autocast(device_type="cuda", dtype=_AUTOCAST_DTYPE[autocast_dtype])


class BaseSolver(nn.Module):
    def __init__(
        self,
        config: PolicySpec,
        env: ResUNSAT,
    ):
        super(BaseSolver, self).__init__()
        self.config = config
        self.env = env
        self.config['env'] = env
        self.lr_schedule = config['lr_schedule']
        self.gamma = config['gamma']
        self.max_grad_norm = config['max_grad_norm']
        self.device = config['device']
        self.all_sat = "OmNeuRes" in config['solver']

        self.config['critic'] = False
        self._setup_model()
        self.load_checkpoint(config['checkpoint'])
        self.apply_freezers(config['freeze'])
        self.setup_losses()
        self.total_episodes = 0
        self.ep_count = 0
        self.reset()
        self.to(config['device'])

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
        # if self.config['finetune_schedule'] is not None:
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

        if self.config['predict_sat']:
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
        # Batched-path buffers — coexist with the single-formula ones above so
        # both code paths can be used in the same process (e.g. tests).
        self.batch_data = []
        self._step_buffers = {"c_logp": [], "sat_loss": [], "sat_vote": [], "valid": []}

    def reset(self, train: bool = True):
        self.reset_buffers()
        self.policy.train(train)
        self.policy.reset()

    def post_episode(self, rewards):
        self.ep_data["ep_len"].append(len(rewards))
        return

    # =====================================================================
    # Batched API. The trainer drives one lockstep step at a time via
    # resolve_batch, then calls finalize_batch when env.all_done(). Every
    # `update_every` batches it calls update_batch.
    # =====================================================================
    def setup_batch(self, formulas, sample_idxs=None):
        self.is_sat_b = th.tensor(
            [f.is_sat for f in formulas], dtype=th.bool, device=self.device
        )
        self.policy.setup_batch(formulas, sample_idxs=sample_idxs)

    def resolve_batch(self, guide_acts):
        """One lockstep step. Returns list[dict] of length B for env.step_batch."""
        with _make_autocast_ctx(self.device, self.config.get('autocast_dtype', 'none')):
            running = self.env.running_mask_t()                      # (B,) bool, before step
            ret = self.policy.forward_batch(guide_acts, running)

            # Done rows still flow through forward_batch; any -inf / NaN they
            # produce in c_logp / sat_loss / sat_vote (e.g. selectors gathering at
            # masked positions) would poison the masked-mean loss as 0 * -inf = NaN.
            # Substitute zeros at done rows so the mask cleanly drops them.
            # Cast to fp32 so loss accumulation in compute_loss_batch stays
            # precise even when autocast is producing bf16/fp16 elsewhere.
            def _zero_at_done(t):
                return th.where(running, t, th.zeros_like(t)).float()
            if "c_logp" in ret:
                self._step_buffers["c_logp"].append(_zero_at_done(ret["c_logp"]))
                self._step_buffers["valid"].append(running)
            if "sat_loss" in ret:
                self._step_buffers["sat_loss"].append(_zero_at_done(ret["sat_loss"]))
            if "sat_vote" in ret:
                self._step_buffers["sat_vote"].append(_zero_at_done(ret["sat_vote"]))

            B = self.env.batch_size_b
            actions = []
            c_idx = ret.get("c_idx")
            if c_idx is None:
                # MP-only step: advance embedder hidden state without adding clauses.
                for b in range(B):
                    a = {"Res": None, "VA": None}
                    if "A_pred" in ret and bool(running[b]):
                        a["VA"] = [pred[b] for pred in ret["A_pred"]]
                    actions.append(a)
                self.policy.perform_mp_round_batch()
                return actions

            c_idx_cpu = c_idx.detach().cpu().numpy()
            new_clauses_per_row = []
            for b in range(B):
                if not bool(running[b]):
                    new_clauses_per_row.append([])
                    actions.append({"Res": None, "VA": None})
                    continue
                i, j = int(c_idx_cpu[b, 0]), int(c_idx_cpu[b, 1])
                new_c = None
                try:
                    c1 = self.env.clauses_b[b][i]
                    c2 = self.env.clauses_b[b][j]
                    new_c = self.env._resolve_row(b, c1, c2)
                except IndexError:
                    pass
                new_clauses_per_row.append([new_c] if new_c is not None else [])
                a = {"Res": (i, j), "VA": None}
                if "A_pred" in ret:
                    a["VA"] = [pred[b] for pred in ret["A_pred"]]
                actions.append(a)
            self.policy.add_new_clauses_batched(new_clauses_per_row, running)
            return actions

    def finalize_batch(self):
        """Snapshot the per-step buffers as one batch entry. Called by trainer
        after env.all_done()."""
        sb = self._step_buffers
        if not sb["c_logp"] and not sb["sat_loss"] and not sb["sat_vote"]:
            return
        snap = {"is_sat": self.is_sat_b}
        if sb["c_logp"]:
            snap["c_logp"] = th.stack(sb["c_logp"], dim=0)        # (T, B)
            snap["valid"] = th.stack(sb["valid"], dim=0)           # (T, B) bool
        if sb["sat_loss"]:
            snap["sat_loss"] = th.stack(sb["sat_loss"], dim=0)
        if sb["sat_vote"]:
            snap["sat_vote"] = th.stack(sb["sat_vote"], dim=0)
        snap["ep_lens"] = th.from_numpy(self.env.ep_len_b.copy()).long().to(self.device)
        self.batch_data.append(snap)
        self._step_buffers = {"c_logp": [], "sat_loss": [], "sat_vote": [], "valid": []}

    def _build_TH(self, ep_lens: th.LongTensor, T: int) -> th.Tensor:
        """For each row b, the first ep_lens[b] timesteps get descending gamma
        weights (matches get_time_horizon for the single-formula path);
        remaining timesteps are zero (and will be masked out anyway)."""
        device = ep_lens.device
        n = ep_lens.clamp(max=T).to(device).unsqueeze(0)         # (1, B)
        t = th.arange(T, device=device).unsqueeze(1)              # (T, 1)
        valid = t < n                                              # (T, B)
        # gamma^(n-1-t) where valid, else 0 — clamp the exponent so the masked-
        # out gamma^negative values don't overflow before being zeroed.
        exp_safe = (n - 1 - t).clamp(min=0).float()                # (T, B)
        return (self.gamma ** exp_safe) * valid.float()

    def compute_loss_batch(self, no_grad: bool = False) -> dict:
        loss_dict = {}
        total_loss: Any = 0.0
        if not self.batch_data:
            return loss_dict

        res_num = th.zeros((), device=self.device)
        res_denom = th.zeros((), device=self.device)
        sat_num = th.zeros((), device=self.device)
        sat_denom = th.zeros((), device=self.device)
        pred_num = th.zeros((), device=self.device)
        pred_denom = th.zeros((), device=self.device)

        for snap in self.batch_data:
            is_sat_b = snap["is_sat"]                              # (B,)
            ep_lens = snap["ep_lens"]                              # (B,)

            if "c_logp" in snap:
                c_logp = snap["c_logp"]                            # (T, B)
                valid = snap["valid"]                              # (T, B)
                T = c_logp.size(0)
                TH = self._build_TH(ep_lens, T)
                unsat_mask = valid & (~is_sat_b).unsqueeze(0)
                if unsat_mask.any():
                    m = unsat_mask.float()
                    res_num = res_num + -(TH * c_logp * m).sum()
                    res_denom = res_denom + m.sum()

            if "sat_loss" in snap:
                sat_loss = snap["sat_loss"]                        # (T, B)
                T = sat_loss.size(0)
                TH = self._build_TH(ep_lens, T)
                if "valid" in snap and snap["valid"].size(0) >= T:
                    valid_t = snap["valid"][:T]
                else:
                    valid_t = th.ones(T, sat_loss.size(1), dtype=th.bool, device=self.device)
                sat_mask = valid_t & is_sat_b.unsqueeze(0)
                if sat_mask.any():
                    m = sat_mask.float()
                    sat_num = sat_num + (TH * sat_loss * m).sum()
                    sat_denom = sat_denom + m.sum()

            if self.config['predict_sat'] and "sat_vote" in snap:
                sat_vote = snap["sat_vote"]                        # (T, B)
                T = sat_vote.size(0)
                if "valid" in snap and snap["valid"].size(0) >= T:
                    valid_t = snap["valid"][:T]
                else:
                    valid_t = th.ones(T, sat_vote.size(1), dtype=th.bool, device=self.device)
                vt = valid_t.float()
                # Per-row mean over valid steps; matches the single-formula
                # behavior of averaging the sat_vote across the episode.
                vt_sum = vt.sum(dim=0).clamp(min=1.0)
                sv_mean = (sat_vote * vt).sum(dim=0) / vt_sum     # (B,)
                target = is_sat_b.float()
                bce = binary_cross_entropy_with_logits(sv_mean, target, reduction='none')
                pred_num = pred_num + bce.sum()
                pred_denom = pred_denom + th.tensor(float(bce.numel()), device=self.device)

        if res_denom.item() > 0:
            res_loss = res_num / res_denom.clamp(min=1)
            loss_dict["res_loss"] = res_loss
            total_loss = total_loss + res_loss
        if sat_denom.item() > 0:
            sat_loss_v = sat_num / sat_denom.clamp(min=1)
            loss_dict["sat_loss"] = sat_loss_v
            total_loss = total_loss + sat_loss_v
        if pred_denom.item() > 0:
            pred_loss = pred_num / pred_denom
            loss_dict["sat_pred"] = pred_loss
            total_loss = total_loss + pred_loss

        loss_dict["total"] = total_loss
        if no_grad:
            for k, v in loss_dict.items():
                if not isinstance(v, float):
                    loss_dict[k] = v.item()
        return loss_dict

    def update_batch(self) -> dict:
        loss_dict = self.compute_loss_batch()
        if not loss_dict or not isinstance(loss_dict.get("total"), th.Tensor):
            self.batch_data = []
            return {k: (v if isinstance(v, float) else float(v)) for k, v in loss_dict.items()}
        total_loss = loss_dict["total"]
        self.optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        if th.isnan(total_loss):
            print("=" * 50)
            print("NaN detected in batched loss. Skipping update.")
            print("=" * 50)
        else:
            self.optimizer.step()
        self.ep_count += 1
        self.update_lr()
        for k, v in loss_dict.items():
            loss_dict[k] = v.item() if isinstance(v, th.Tensor) else float(v)
        self.batch_data = []
        return loss_dict

    def _setup_model(self):
        self.policy = NeuRes(self.config)
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

        if self.config['predict_sat']:
            sat_pred = th.sigmoid(ret_dict["sat_vote"])
            action["sat_pred"] = int(th.round(sat_pred).item())

        return action
    
    def forward(self, guide_act=None):
        guide_act={"Res": None, "VA": None}
        return self.resolve(guide_act)
