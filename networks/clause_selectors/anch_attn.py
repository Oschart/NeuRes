import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from common.utils import mask_grid, mask_grid_asym
from envs import ResUNSAT
import itertools
import numpy as np
import wandb

class AnchAttention(nn.Module):
	"""This module performs attention to pick the variable to do resolution on, then the clause pair"""
	def __init__(self, opt):
		super(AnchAttention, self).__init__()
		self.opt = opt
		self.env: ResUNSAT = opt.env
		hidden_size = opt.hidden_size
		self.var_K = nn.Linear(hidden_size, hidden_size)
		self.var_Q = nn.Linear(hidden_size, hidden_size)
		self.var_attn = nn.Linear(hidden_size, 1)

		self.W_Q = nn.Linear(hidden_size, hidden_size)
		self.W_K = nn.Linear(hidden_size, hidden_size)

	def select_var(self, state):
		# Keys
		n_vars = state["literal_emb"].shape[1]//2
		L_emb = state["literal_emb"]
		K = L_emb[:, :n_vars]# + L_emb[:, n_vars:]
		K_transform = self.var_K(K)
		if self.opt.C_aggr == "sum":
			Q = state["clause_emb"].sum(dim=1)
		else:
			Q = state["clause_emb"].mean(dim=1)
		Q_transform = self.var_Q(Q)

		u_i = self.var_attn(th.tanh(K_transform + Q_transform)).squeeze(-1)

		log_score = th.nn.functional.log_softmax(u_i, dim=-1)

		if self.expert_pair is not None:
			var_idx = self.env.which_res_var(*self.expert_pair)
			var_logp = log_score[:, var_idx]
		else:
			var_logp, var_idx = th.max(log_score, dim=-1)
			var_idx = var_idx.item()

		if th.isnan(var_logp).any():
			text = f"var_logp: {var_logp}, u_i={u_i}"
			wandb.alert(title="Var Logp is NaN", text=text)

		return var_logp, var_idx

	def get_grid_mask(self):
		res_mask = self.opt.env.res_mask
		return res_mask
	
	
	def select_res_pair(self, logp_grid_exc, expert_pair):
		# Decide which pair to take
		if expert_pair is not None:
			C_logp = logp_grid_exc[:, expert_pair[0], expert_pair[1]]
			C_idx = expert_pair
		else:
			# Get maximal element and index
			N, M = logp_grid_exc.shape[-2:]
			C_logp, C_idx = th.max(logp_grid_exc.flatten(-2, -1), dim=-1)
			# Reshape index to 2D
			C_idx = (C_idx // M, C_idx % M)
			C_idx = (C_idx[0].item(), C_idx[1].item())
		return C_logp, C_idx


	def forward(self, state, expert_pair=None):
		self.expert_pair = expert_pair

		# Select variable
		var_logp, var_idx = self.select_var(state)

		# Pivot by variable
		piv_dict = self.env.pivot_by_var(var_idx)
		pos_idx = list(piv_dict["v_pos_idx"])
		neg_idx = list(piv_dict["v_neg_idx"])
		all_idx = pos_idx + neg_idx

		pool = state["clause_emb"]

		Q_pool = pool
		K_pool = pool

		Q = Q_pool[:, piv_dict["v_pos_idx"]]
		K = K_pool[:, piv_dict["v_neg_idx"]]

		# Compute attention scores
		q = self.W_Q(Q)
		k = self.W_K(K)
		
		attn_scores = th.bmm(q, k.transpose(-2, -1)) / np.sqrt(self.opt.hidden_size)

		keep_mask = piv_dict["v_res_mask"]
		# Remap taken indices to smaller grid
		taken_set_small = set()
		for p in state["taken_set"]:
			# Both indices need to be in all_idx
			if p[0] in all_idx and p[1] in all_idx:
				p_small = [None, None]
				for i in range(2):
					if p[i] in pos_idx:
						p_small[0] = pos_idx.index(p[i])
					else:
						p_small[1] = neg_idx.index(p[i])
				if None in p_small:
					continue
				p_small = tuple(p_small)
				taken_set_small.add(p_small) 


		logp_grid_exc = mask_grid_asym(attn_scores.clone(), taken_set_small, keep_mask=keep_mask)

		# Map expert pair to smaller grid: get index of expert pair in small index arrays
		if expert_pair is not None:
			expert_pair_small = [None, None]
			for i in range(2):
				if expert_pair[i] in pos_idx:
					expert_pair_small[0] = pos_idx.index(expert_pair[i])
				else:
					expert_pair_small[1] = neg_idx.index(expert_pair[i])
			expert_pair_small = tuple(expert_pair_small)
		else:
			expert_pair_small = None

		C_logp, C_idx = self.select_res_pair(logp_grid_exc, expert_pair_small)

		if th.isnan(C_logp).any():
			text = f"C_logp: {C_logp}, logp_grid_exc={logp_grid_exc}"
			wandb.alert(title="Clause Logp is NaN", text=text)

		C_logp = C_logp + var_logp

		# Remap index to original index
		C_idx = (piv_dict["v_pos_idx"][C_idx[0]], piv_dict["v_neg_idx"][C_idx[1]])
		C_idx = [C_idx]
		
		
		state["taken_set"].add(C_idx[0])

		ret_dict = {
			"c_logp": C_logp,
			"c_idx": C_idx,
		}

		return ret_dict