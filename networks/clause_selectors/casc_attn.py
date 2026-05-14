import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from common.utils import Scheduler
import itertools
import numpy as np
from common.utils import masked_log_softmax


class CascAttention(nn.Module):
	def __init__(self, config):
		super(CascAttention, self).__init__()
		self.config = config
		emb_size = config['emb_size']
		self.W1 = nn.Linear(emb_size, emb_size)
		Q_size = emb_size if config['step_attn_cond'] == "add" else 2*emb_size
		self.W2 = nn.Linear(Q_size, emb_size)
		self.vt = nn.Linear(emb_size, 1)

	def step(self, candidates, query, valid_mask=None):
		# (batch_size, max_seq_len, hidden_size)
		key_transform = self.W1(candidates)

		# (batch_size, 1 (unsqueezed), hidden_size)
		query_transform = self.W2(query).unsqueeze(1)

		# 1st line of Eq.(3) in the paper
		# (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
		u_i = self.vt(th.tanh(key_transform + query_transform)).squeeze(-1)

		# softmax with only valid inputs, excluding zero padded parts
		# log-softmax for a better numerical stability
		mask_t = th.ones_like(u_i)
		if valid_mask is not None:
			mask_t[:, ~valid_mask] = 0.0
		
		log_score = masked_log_softmax(u_i, mask_t, dim=-1)

		return log_score

	def forward(self, state, expert_pair=None):
		pool = state["clause_emb"]

		K = pool
		# Q = th.zeros_like(pool[:, 0])
		if self.config['C_aggr'] == "sum":
			mean_clause = pool.sum(dim=1)
		else:
			mean_clause = pool.mean(dim=1)
		if self.config['step_attn_cond'] == "add":
			Q = mean_clause
		else:
			# Concat with zero vector
			Q = th.cat([mean_clause, th.zeros_like(mean_clause)], dim=-1)
		log_pointer_score1 = self.step(K, Q)
		# Maximal index
		if expert_pair is None:
			index1 = th.argmax(log_pointer_score1, dim=-1).squeeze(0)
			index1 = int(index1)
		else:
			index1 = expert_pair[0]
		
		c1_key = K[:, index1]
		# Condition hidden state on the selected clause
		if self.config['step_attn_cond'] == "add":
			Q = c1_key + mean_clause
		else:
			Q = th.cat([mean_clause, c1_key], dim=-1)
		state["taken_map"][index1] = state["taken_map"].get(index1, []) 
		# Construct mask
		res_mask = self.config['env'].res_mask
		mask = res_mask[index1]
		mask[state["taken_map"][index1]] = False

		log_pointer_score2 = self.step(K, Q, mask)
		# Maximal index
		if expert_pair is None:
			index2 = th.argmax(log_pointer_score2, dim=-1).squeeze(0)
			index2 = int(index2)
		else:
			index2 = expert_pair[1]

		max_logp = log_pointer_score1[:, index1] + log_pointer_score2[:, index2]
		max_idx = [(index1, index2)]

		state["taken_map"][index1] = state["taken_map"].get(index1, []) + [index2]
		state["taken_map"][index2] = state["taken_map"].get(index2, []) + [index1]

		ret_dict = {
			"c_logp": max_logp,
			"c_idx": max_idx,
		}

		return ret_dict

	def forward_batch(self, state, expert_pairs=None):
		"""Batched cascaded attention. State carries:
		  clause_emb:     (B, C, emb)
		  clause_mask:    (B, C) bool (True = real clause)
		  res_mask_pair:  (B, C, C) bool (True = pair currently resolvable)
		  taken_map:      list[dict] of length B (mutated in place)
		"""
		pool = state["clause_emb"]                              # (B, C, emb)
		B, C, _ = pool.shape
		device = pool.device
		clause_mask = state["clause_mask"][:, :C]                # (B, C)
		res_mask_pair = state["res_mask_pair"]                   # (B, C, C)

		K = pool
		if self.config['C_aggr'] == "sum":
			mean_clause = pool.sum(dim=1)
		else:
			mean_clause = pool.mean(dim=1)
		if self.config['step_attn_cond'] == "add":
			Q = mean_clause
		else:
			Q = th.cat([mean_clause, th.zeros_like(mean_clause)], dim=-1)

		log_p1 = self.step(K, Q)                                 # (B, C)
		# Mask invalid clauses for first pick
		mask1 = clause_mask.float()
		mask1_safe = mask1.clone()
		all_invalid1 = mask1.sum(dim=-1) == 0
		if all_invalid1.any():
			mask1_safe[all_invalid1, 0] = 1.0
		log_p1 = masked_log_softmax(log_p1, mask1_safe, dim=-1)

		if expert_pairs is None:
			index1 = th.argmax(log_p1, dim=-1)                  # (B,)
		else:
			index1 = th.tensor(
				[p[0] if p is not None else 0 for p in expert_pairs],
				dtype=th.long, device=device,
			)

		# Gather c1_key per row: (B, emb)
		c1_key = th.gather(K, 1, index1.view(B, 1, 1).expand(B, 1, K.size(-1))).squeeze(1)
		if self.config['step_attn_cond'] == "add":
			Q2 = c1_key + mean_clause
		else:
			Q2 = th.cat([mean_clause, c1_key], dim=-1)

		# Build per-row second-step mask: res_mask_pair[b, index1[b]] AND not in taken_map.
		mask2 = th.gather(
			res_mask_pair, 1,
			index1.view(B, 1, 1).expand(B, 1, C)
		).squeeze(1).float()                                     # (B, C)
		idx1_cpu = index1.detach().cpu().tolist()
		taken_map = state["taken_map"]
		for b in range(B):
			i1 = idx1_cpu[b]
			taken = taken_map[b].get(i1, [])
			if taken:
				mask2[b, taken] = 0.0
			# Always exclude self-resolution explicitly
			mask2[b, i1] = 0.0

		log_p2 = self.step(K, Q2)                                # (B, C)
		mask2_safe = mask2.clone()
		all_invalid2 = mask2.sum(dim=-1) == 0
		if all_invalid2.any():
			mask2_safe[all_invalid2, 0] = 1.0
		log_p2 = masked_log_softmax(log_p2, mask2_safe, dim=-1)

		if expert_pairs is None:
			index2 = th.argmax(log_p2, dim=-1)
		else:
			index2 = th.tensor(
				[p[1] if p is not None else 0 for p in expert_pairs],
				dtype=th.long, device=device,
			)

		# logp_1[b, index1[b]] + logp_2[b, index2[b]]
		idx1 = index1.unsqueeze(-1)
		idx2 = index2.unsqueeze(-1)
		c_logp = (
			th.gather(log_p1, 1, idx1).squeeze(-1)
			+ th.gather(log_p2, 1, idx2).squeeze(-1)
		)
		c_idx = th.stack([index1, index2], dim=-1)               # (B, 2)

		# Per-row taken_map update.
		running_mask = state.get("running_mask")
		for b in range(B):
			if running_mask is not None and not bool(running_mask[b]):
				continue
			i1, i2 = idx1_cpu[b], int(index2[b].item())
			taken_map[b][i1] = taken_map[b].get(i1, []) + [i2]
			taken_map[b][i2] = taken_map[b].get(i2, []) + [i1]

		return {"c_logp": c_logp, "c_idx": c_idx}
