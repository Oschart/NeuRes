import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from common.utils import mask_grid, mask_grid_asym, masked_log_softmax
from envs import ResUNSAT
import itertools
import numpy as np
import wandb

class AnchAttention(nn.Module):
	"""This module performs attention to pick the variable to do resolution on, then the clause pair"""
	def __init__(self, config):
		super(AnchAttention, self).__init__()
		self.config = config
		self.env: ResUNSAT = config['env']
		emb_size = config['emb_size']
		self.var_K = nn.Linear(emb_size, emb_size)
		self.var_Q = nn.Linear(emb_size, emb_size)
		self.var_attn = nn.Linear(emb_size, 1)

		self.W_Q = nn.Linear(emb_size, emb_size)
		self.W_K = nn.Linear(emb_size, emb_size)

	def select_var(self, state):
		# Keys
		n_vars = state["literal_emb"].shape[1]//2
		L_emb = state["literal_emb"]
		K = L_emb[:, :n_vars]
		K_transform = self.var_K(K)
		if self.config['C_aggr'] == "sum":
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

		Q = pool[:, piv_dict["v_pos_idx"]]
		K = pool[:, piv_dict["v_neg_idx"]]

		# Compute attention scores
		q = self.W_Q(Q)
		k = self.W_K(K)
		
		attn_scores = th.bmm(q, k.transpose(-2, -1)) / np.sqrt(self.config['emb_size'])

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

	# Batched API ---------------------------------------------------------
	def select_var_batch(self, state, expert_pairs=None):
		"""Returns (var_logp: (B,), var_idx: (B,)). For supervised steps the
		variable is determined by the expert pair via env.which_res_var_batched."""
		L_emb = state["literal_emb"]                            # (B, max_lits, emb)
		var_mask = state["var_mask"]                             # (B, max_vars)
		max_vars = var_mask.size(1)
		K = L_emb[:, :max_vars]                                  # positive literal slice
		K_t = self.var_K(K)
		if self.config['C_aggr'] == "sum":
			Q = state["clause_emb"].sum(dim=1)
		else:
			Q = state["clause_emb"].mean(dim=1)
		Q_t = self.var_Q(Q).unsqueeze(1)                          # (B, 1, emb)
		u_i = self.var_attn(th.tanh(K_t + Q_t)).squeeze(-1)       # (B, max_vars)
		# Mask padded vars to -inf before log_softmax. Rows with no real vars
		# (shouldn't happen in practice) get a fake unmasked entry at 0 to
		# avoid all-nan softmax; their logp is meaningless and gets masked
		# downstream.
		mask = var_mask.float()
		all_invalid = mask.sum(dim=-1) == 0
		mask_safe = mask
		if all_invalid.any():
			mask_safe = mask.clone()
			mask_safe[all_invalid, 0] = 1.0
		log_score = masked_log_softmax(u_i, mask_safe, dim=-1)    # (B, max_vars)

		B = L_emb.size(0)
		device = L_emb.device
		if expert_pairs is None:
			var_logp, var_idx = th.max(log_score, dim=-1)         # (B,), (B,)
		else:
			# Look up the pivot variable per row from the env helper.
			env = self.config['env']
			i = th.tensor(
				[p[0] if p is not None else 0 for p in expert_pairs],
				dtype=th.long, device=device,
			)
			j = th.tensor(
				[p[1] if p is not None else 0 for p in expert_pairs],
				dtype=th.long, device=device,
			)
			var_idx = env.which_res_var_batched(i, j)             # (B,)
			var_idx = var_idx.clamp(0, max_vars - 1)
			var_logp = th.gather(log_score, 1, var_idx.unsqueeze(-1)).squeeze(-1)
		return var_logp, var_idx

	def forward_batch(self, state, expert_pairs=None):
		assert self.config.get('topk', 1) == 1, \
			"Batched AnchAttention currently only supports topk=1"
		var_logp, var_idx = self.select_var_batch(state, expert_pairs)

		env = self.config['env']
		piv = env.pivot_by_var_batched(var_idx)
		pos_idx = piv["pos_idx"]                                 # (B, max_pos)
		neg_idx = piv["neg_idx"]                                 # (B, max_neg)
		pos_mask = piv["pos_mask"]                                # (B, max_pos)
		neg_mask = piv["neg_mask"]                                # (B, max_neg)
		v_res_mask = piv["v_res_mask"]                            # (B, max_pos, max_neg)
		max_pos = piv["max_pos"]
		max_neg = piv["max_neg"]

		pool = state["clause_emb"]                                # (B, C, emb)
		B, C, emb = pool.shape
		device = pool.device

		# Gather Q/K from clause_emb at the (padded) pos/neg indices.
		Q = th.gather(pool, 1, pos_idx.unsqueeze(-1).expand(B, max_pos, emb))
		K = th.gather(pool, 1, neg_idx.unsqueeze(-1).expand(B, max_neg, emb))
		q = self.W_Q(Q)
		k = self.W_K(K)
		attn_scores = th.bmm(q, k.transpose(-2, -1)) / np.sqrt(self.config['emb_size'])  # (B, max_pos, max_neg)

		# Build per-row taken_pair_mask: True where the (global_pos, global_neg)
		# pair is already taken. For each (p, n) in the small grid, look up
		# global indices via pos_idx[b, p] / neg_idx[b, n] and test against
		# state["taken_set"][b].
		taken_set_list = state["taken_set"]
		# Bring small-grid global indices to CPU once for the lookup loop.
		pos_idx_cpu = pos_idx.detach().cpu().numpy()
		neg_idx_cpu = neg_idx.detach().cpu().numpy()
		pos_mask_cpu = pos_mask.detach().cpu().numpy()
		neg_mask_cpu = neg_mask.detach().cpu().numpy()
		# Per-row global→small lookup tables (only for valid slots), built
		# once and reused by both the taken-set scan and the expert-pair scan.
		g2pos_list = [None] * B
		g2neg_list = [None] * B
		def _row_lookup(b):
			if g2pos_list[b] is None:
				pl = int(pos_mask_cpu[b].sum())
				nl = int(neg_mask_cpu[b].sum())
				g2pos_list[b] = {int(pos_idx_cpu[b, p]): p for p in range(pl)}
				g2neg_list[b] = {int(neg_idx_cpu[b, n]): n for n in range(nl)}
			return g2pos_list[b], g2neg_list[b]
		taken_pair_mask_np = np.zeros((B, max_pos, max_neg), dtype=bool)
		for b in range(B):
			ts = taken_set_list[b]
			if not ts:
				continue
			g2pos, g2neg = _row_lookup(b)
			for (p_glob, n_glob) in ts:
				ps = g2pos.get(int(p_glob))
				ns = g2neg.get(int(n_glob))
				if ps is not None and ns is not None:
					taken_pair_mask_np[b, ps, ns] = True
				# Also try the swapped role (since the original code's add
				# stored the raw (i, j) without canonicalizing pos vs neg).
				ps2 = g2pos.get(int(n_glob))
				ns2 = g2neg.get(int(p_glob))
				if ps2 is not None and ns2 is not None:
					taken_pair_mask_np[b, ps2, ns2] = True
		taken_pair_mask = th.from_numpy(taken_pair_mask_np).to(device)

		valid = (
			v_res_mask
			& ~taken_pair_mask
			& pos_mask.unsqueeze(-1)
			& neg_mask.unsqueeze(1)
		)                                                          # (B, max_pos, max_neg)

		flat_attn = attn_scores.reshape(B, max_pos * max_neg)
		flat_valid = valid.reshape(B, max_pos * max_neg)
		all_invalid = ~flat_valid.any(dim=-1)
		if all_invalid.any():
			flat_valid = flat_valid.clone()
			flat_valid[all_invalid, 0] = True
		flat_attn = flat_attn.masked_fill(~flat_valid, float('-inf'))
		flat_logp = th.nn.functional.log_softmax(flat_attn, dim=-1)

		if expert_pairs is None:
			max_logp, flat_idx = flat_logp.max(dim=-1)
			ps = flat_idx // max_neg
			ns = flat_idx % max_neg
		else:
			# Map the expert (i, j) global pair into the small (ps, ns) grid.
			ps_list, ns_list = [], []
			for b in range(B):
				p = expert_pairs[b]
				if p is None:
					ps_list.append(0); ns_list.append(0)
					continue
				pi, pj = int(p[0]), int(p[1])
				g2pos, g2neg = _row_lookup(b)
				ps_b = g2pos.get(pi)
				ns_b = g2neg.get(pj)
				if ps_b is None or ns_b is None:
					# Try swapped role
					ps_b = g2pos.get(pj)
					ns_b = g2neg.get(pi)
				if ps_b is None or ns_b is None:
					ps_b = 0; ns_b = 0
				ps_list.append(ps_b); ns_list.append(ns_b)
			ps = th.tensor(ps_list, dtype=th.long, device=device)
			ns = th.tensor(ns_list, dtype=th.long, device=device)
			max_logp = flat_logp[th.arange(B, device=device), ps * max_neg + ns]

		# Map small grid indices back to global clause indices.
		c_pos_global = th.gather(pos_idx, 1, ps.unsqueeze(-1)).squeeze(-1)
		c_neg_global = th.gather(neg_idx, 1, ns.unsqueeze(-1)).squeeze(-1)
		c_logp = max_logp + var_logp                              # (B,)
		c_idx = th.stack([c_pos_global, c_neg_global], dim=-1)    # (B, 2)

		# Per-row taken_set update: only running rows.
		running_mask = state.get("running_mask")
		c_idx_cpu = c_idx.detach().cpu().tolist()
		for b in range(B):
			if running_mask is not None and not bool(running_mask[b]):
				continue
			taken_set_list[b].add((int(c_idx_cpu[b][0]), int(c_idx_cpu[b][1])))

		return {"c_logp": c_logp, "c_idx": c_idx}