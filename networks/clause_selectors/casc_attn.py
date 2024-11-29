import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from common.utils import Scheduler
import itertools
import numpy as np
from common.utils import masked_log_softmax


class CascAttention(nn.Module):
	def __init__(self, opt):
		super(CascAttention, self).__init__()
		self.opt = opt
		hidden_size = opt.hidden_size
		self.hidden_size = hidden_size
		self.W1 = nn.Linear(hidden_size, hidden_size)
		Q_size = hidden_size if opt.step_attn_cond == "add" else 2*hidden_size
		self.W2 = nn.Linear(Q_size, hidden_size)
		self.vt = nn.Linear(hidden_size, 1)

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
		if self.opt.C_aggr == "sum":
			mean_clause = pool.sum(dim=1)
		else:
			mean_clause = pool.mean(dim=1)
		if self.opt.step_attn_cond == "add":
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
		if self.opt.step_attn_cond == "add":
			Q = c1_key + mean_clause
		else:
			Q = th.cat([mean_clause, c1_key], dim=-1)
		state["taken_map"][index1] = state["taken_map"].get(index1, []) 
		# Construct mask
		res_mask = self.opt.env.res_mask
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
	
	