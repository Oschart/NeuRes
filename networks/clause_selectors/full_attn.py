import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from common.utils import mask_grid
import itertools
import numpy as np
import wandb
import math


class FullAttention(nn.Module):
	"""This module performs cross-attention between elements of the same sequence"""
	def __init__(self, config):
		super(FullAttention, self).__init__()
		self.config = config
		hidden_size = config['hidden_size']
		q_width = hidden_size
		self.W_Q = nn.Linear(q_width, hidden_size)
		self.W_K = nn.Linear(hidden_size, hidden_size)
		# Cached upper-triangle bool mask, grown on demand and sliced per call.
		self._upper_cache: Optional[th.Tensor] = None

	def _get_upper(self, C: int, device) -> th.Tensor:
		cache = self._upper_cache
		if cache is None or cache.size(0) < C or cache.device != device:
			new_size = max(C, cache.size(0) * 2 if cache is not None else C)
			self._upper_cache = th.triu(
				th.ones(new_size, new_size, device=device, dtype=th.bool),
				diagonal=1,
			)
		return self._upper_cache[:C, :C]


	def get_grid_mask(self):
		res_mask = self.config['env'].res_mask
		res_mask = th.from_numpy(res_mask)
		# res_mask = th.as_tensor(res_mask, device=self.config['device'])
		return res_mask
	
	def select_res_pair(self, logp_grid_exc, expert_pair):
		# Decide which pair to take
		if expert_pair is not None:
			logp1 = logp_grid_exc[:, expert_pair[0], expert_pair[1]]
			logp2 = logp_grid_exc[:, expert_pair[1], expert_pair[0]]
			max_logp = th.max(logp1, logp2)
			max_idx = expert_pair
		else:
			# Get maximal element and index
			N = logp_grid_exc.shape[-1]
			max_logp, max_idx = th.max(logp_grid_exc.flatten(-2, -1), dim=-1)
			# Reshape index to 2D
			max_idx = (max_idx // N, max_idx % N)
			max_idx = (max_idx[0].item(), max_idx[1].item())
		return max_logp, max_idx

	
	# @profile
	def efficient_attn3(self, state, q, k, keep_mask, expert_pair):
		# Efficient attention:
		# Mask out lower triangle of attention matrix
		mask_upper = th.triu(keep_mask, diagonal=1).to(self.config['device'])

		R, C = th.where(mask_upper)
		attn_scores = th.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.config['hidden_size'])

		if self.config['mask_mode'] == "upper+lower":
			mask_lower = th.tril(keep_mask, diagonal=-1).to(self.config['device'])
			attn_scores = attn_scores[:, mask_upper] + attn_scores[:, mask_lower]
		elif self.config['mask_mode'] == "max(upper,lower)":
			mask_lower = th.tril(keep_mask, diagonal=-1).to(self.config['device'])
			attn_scores = th.max(attn_scores[:, mask_upper], attn_scores[:, mask_lower])
		elif self.config['mask_mode'] == "min(upper,lower)":
			mask_lower = th.tril(keep_mask, diagonal=-1).to(self.config['device'])
			attn_scores = th.min(attn_scores[:, mask_upper], attn_scores[:, mask_lower])
		else: # upper
			attn_scores = attn_scores[:, mask_upper]

		logp_scores = th.nn.functional.log_softmax(attn_scores, dim=-1)
		if expert_pair is None:
			topk = th.topk(logp_scores, k=self.config['topk'], dim=-1)
			sel_logp, sel_idx = topk.values[0], topk.indices[0]
			sel_idx = sel_idx.cpu()
			sel_idx = [(R[x].item(), C[x].item()) for x in sel_idx]
		else:
			# Get logp for expert pair
			sel_idx = tuple(sorted(expert_pair))
			# Remap to 1D
			sel_idx_1d = th.where((R == sel_idx[0]) & (C == sel_idx[1]))[0][0]
			sel_logp = logp_scores[:, sel_idx_1d]
			sel_idx = [sel_idx]
			if self.config['topk_train'] and self.config['topk'] > 1:
				topk = th.topk(logp_scores, k=self.config['topk'], dim=-1)
				_, sel_idx2 = topk.values[0], topk.indices[0]
				sel_idx2 = sel_idx2.cpu()
				sel_idx2 = [(R[x].item(), C[x].item()) for x in sel_idx2]
				sel_idx = sel_idx + sel_idx2

		return sel_logp, sel_idx
	

	# @profile
	def forward(self, state, expert_pair=None):
		# Project query, key, and value
		pool = state["clause_emb"]
		Q = pool
		K = pool

		# Compute attention scores
		q = self.W_Q(Q)
		k = self.W_K(K)

		keep_mask = self.get_grid_mask()

		sel_logp, sel_idx = self.efficient_attn3(state, q, k, keep_mask, expert_pair)

		ret_dict = {
			"c_logp": sel_logp,
			"c_idx": sel_idx,
		}

		return ret_dict

	# Batched variant. Inputs are (B,...); returns per-row tensors.
	# Only mask_mode="upper" is supported in the batched path; the lower-triangle
	# folding modes (upper+lower / max / min) would require per-row sparse
	# indexing that doesn't vectorize cleanly. Other modes raise.
	def forward_batch(self, state, expert_pairs=None):
		assert self.config.get('mask_mode', 'upper') == 'upper', \
			"Batched FullAttention currently only supports mask_mode='upper'"
		assert self.config.get('topk', 1) == 1, \
			"Batched FullAttention currently only supports topk=1"

		pool = state["clause_emb"]                              # (B, C, emb)
		B, C, _ = pool.shape
		device = pool.device
		q = self.W_Q(pool)
		k = self.W_K(pool)
		# (B, C, C)
		attn = th.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.config['hidden_size'])

		keep_mask = state["res_mask_pair"]                      # (B, C, C) bool
		clause_mask = state["clause_mask"][:, :C]                # (B, C) bool
		pair_valid = (
			keep_mask
			& clause_mask.unsqueeze(1)
			& clause_mask.unsqueeze(2)
		)
		upper = self._get_upper(C, device)
		pair_valid = pair_valid & upper.unsqueeze(0)

		flat_attn = attn.reshape(B, C * C)
		flat_valid = pair_valid.reshape(B, C * C)
		# Rows with no valid pair (e.g., already-done rows): give them a fake
		# valid pair at (0,0) so log_softmax doesn't return all -inf / nan. The
		# resulting c_logp is meaningless but will be masked out by the solver.
		all_invalid = ~flat_valid.any(dim=-1)                   # (B,)
		if all_invalid.any():
			flat_valid = flat_valid.clone()
			flat_valid[all_invalid, 0] = True
		flat_attn = flat_attn.masked_fill(~flat_valid, float('-inf'))
		flat_logp = th.nn.functional.log_softmax(flat_attn, dim=-1)  # (B, C*C)

		if expert_pairs is None:
			max_logp, flat_idx = flat_logp.max(dim=-1)
			i = flat_idx // C
			j = flat_idx % C
		else:
			# Per-row gather at expert positions (sorted to honor mask_mode='upper').
			# For rows with no expert pair (e.g. already-done), gather at (0,0)
			# and overwrite with 0.0 so that subsequent loss masking (which
			# multiplies by 0) does not produce 0*-inf=NaN.
			ij_np = np.zeros((B, 2), dtype=np.int64)
			has_expert_np = np.zeros(B, dtype=bool)
			any_missing = False
			for b, p in enumerate(expert_pairs):
				if p is None:
					any_missing = True
					continue
				a, c_ = p if p[0] <= p[1] else (p[1], p[0])
				ij_np[b, 0] = a
				ij_np[b, 1] = c_
				has_expert_np[b] = True
			ij = th.from_numpy(ij_np).to(device)
			i, j = ij[:, 0], ij[:, 1]
			gathered = flat_logp[th.arange(B, device=device), i * C + j]
			if any_missing:
				safe = th.from_numpy(has_expert_np).to(device)
				max_logp = th.where(safe, gathered, th.zeros_like(gathered))
			else:
				max_logp = gathered

		c_idx = th.stack([i, j], dim=-1)                         # (B, 2)
		return {
			"c_logp": max_logp,                                  # (B,)
			"c_idx": c_idx,
		}