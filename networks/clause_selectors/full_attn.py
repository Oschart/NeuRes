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
	def __init__(self, opt):
		super(FullAttention, self).__init__()
		self.opt = opt
		hidden_size = opt.hidden_size
		q_width = hidden_size
		self.W_Q = nn.Linear(q_width, hidden_size)
		self.W_K = nn.Linear(hidden_size, hidden_size)


	def get_grid_mask(self):
		res_mask = self.opt.env.res_mask
		res_mask = th.from_numpy(res_mask)
		# res_mask = th.as_tensor(res_mask, device=self.opt.device)
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
		mask_upper = th.triu(keep_mask, diagonal=1).to(self.opt.device)

		R, C = th.where(mask_upper)
		attn_scores = th.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.opt.hidden_size)

		if self.opt.mask_mode == "upper+lower":
			mask_lower = th.tril(keep_mask, diagonal=-1).to(self.opt.device)
			attn_scores = attn_scores[:, mask_upper] + attn_scores[:, mask_lower]
		elif self.opt.mask_mode == "max(upper,lower)":
			mask_lower = th.tril(keep_mask, diagonal=-1).to(self.opt.device)
			attn_scores = th.max(attn_scores[:, mask_upper], attn_scores[:, mask_lower])
		elif self.opt.mask_mode == "min(upper,lower)":
			mask_lower = th.tril(keep_mask, diagonal=-1).to(self.opt.device)
			attn_scores = th.min(attn_scores[:, mask_upper], attn_scores[:, mask_lower])
		else: # upper
			attn_scores = attn_scores[:, mask_upper]

		logp_scores = th.nn.functional.log_softmax(attn_scores, dim=-1)
		if expert_pair is None:
			topk = th.topk(logp_scores, k=self.opt.topk, dim=-1)
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
			if self.opt.topk_train and self.opt.topk > 1:
				topk = th.topk(logp_scores, k=self.opt.topk, dim=-1)
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