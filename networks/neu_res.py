import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from common.utils import Scheduler
import itertools
import numpy as np
from networks.embedders.msg_passing import MP_Embedder, MLP
from networks.basic_nets import Encoder, Critic, AssignDecoder
from networks.clause_selectors import make_attn_module
from common.problem import Formula
from envs import ResUNSAT


def zero_net():
	zero_fn = lambda x: 0
	return zero_fn


class NeuRes(nn.Module):
	def __init__(self, opt):
		super(NeuRes, self).__init__()

		self.opt = opt
		self.env: ResUNSAT = opt.env
		# Embedding dimension
		self.emb_size = opt.emb_size
		# (Decoder) hidden size
		self.hidden_size = opt.hidden_size
		# Bidirectional Encoder
		self.bidirectional = opt.bidirectional
		self.num_directions = 2 if opt.bidirectional else 1
		self.enc_dec_cycle = opt.enc_dec_cycle
		self.num_layers = opt.num_layers
		self.eps_greed = 0.0
		self.device = opt.device
		self.reset()

		self.embedder = MP_Embedder(opt)
		self.c_selector = make_attn_module(opt.c_selector, opt)

		if opt.predict_sat:
			self.sat_vote = MLP(opt.emb_size, opt.emb_size, 1)
			self.vote_bias = nn.Parameter(th.zeros(1))


		if opt.critic:
			self.critic = Critic(opt)
		else:
			self.critic = zero_net()

		if "sat" in opt.dataset or "OmNeuRes" in opt.solver:
			self.assign_decoder = AssignDecoder(opt)
		else:
			self.assign_decoder = None
			

		for m in self.modules():
			if isinstance(m, nn.Linear):
				if m.bias is not None:
					th.nn.init.zeros_(m.bias)
		self.to(self.device)

	def reset(self):
		self.state = {
			"clause_emb": None,
			"literal_emb": None,
			"mask": [],
			"taken_map": {},
			"taken_set": set(),
			"dec_input": None, 
			"dec_hidden": None,
			"last_var": None,
			"last_logp": None
		}

	def perform_mp_round(self):
		self.embedder.perform_full_round()
		literal_embeddings = self.embedder.L_state[0]
		clause_embeddings = self.embedder.C_state[0]

		self.state["clause_emb"] = clause_embeddings
		self.state["literal_emb"] = literal_embeddings
		return
	
	def add_new_clause(self, idx_pairs):
		new_clauses = []
		for idx_pair in idx_pairs:
			(c1, c2) = self.env.clauses[idx_pair[0]], self.env.clauses[idx_pair[1]]
			c_new = self.env.resolve(c1, c2)
			if c_new is None or len(c_new) == 0 or c_new in new_clauses:
				continue
			new_clauses.append(c_new)
		emb_ret = self.embedder.embed_clause(new_clauses)
		new_emb = emb_ret["new_clause_emb"]
		
		if self.opt.attn_QK == "Emb_reuse":
			self.state["clause_emb"] = th.cat([self.state["clause_emb"], new_emb.unsqueeze(0)], dim=1)
		else: # Emb
			self.state["clause_emb"] = emb_ret["clauses"].unsqueeze(0)
			self.state["literal_emb"] = emb_ret["literals"].unsqueeze(0)
			
		return
	
	def embed(self, formula: Formula):
		F_emb = self.embedder.init(formula)
		L_emb = F_emb["literals"].unsqueeze(0)
		C_emb = F_emb["clauses"].unsqueeze(0)
		
		self.state["literal_emb"] = L_emb
		self.state["clause_emb"] = C_emb
		return C_emb


	def setup_formula(self, formula: Formula):
		# self.embedder.reset()
		self.reset()
		self.formula = formula
		self.is_sat = formula.is_sat
		self.embed(formula)


	def forward(self, expert_action = None):

		ret_dict = {}
		# SAT votes
		if self.opt.predict_sat:
			L_emb = self.state["literal_emb"]
			# n_vars = L_emb.shape[1]//2
			sat_votes = self.sat_vote(L_emb)
			ret_dict["sat_vote"] = th.mean(sat_votes) + self.vote_bias

		is_sat_train = expert_action is not None and expert_action["VA"] is not None
		unsupervised = not self.env.supervised
		if self.assign_decoder is not None and (is_sat_train or unsupervised):
			a_dict = self.assign_decoder(self.state, expert_action["VA"])
			ret_dict = {**ret_dict, **a_dict}

		if self.is_sat and not self.opt.res_sat_loss:
			if self.opt.no_res:
				ret_dict_c = {}
			else:
				with th.no_grad():
					ret_dict_c = self.c_selector(self.state, expert_pair=expert_action["Res"])
		else:
			ret_dict_c = self.c_selector(self.state, expert_pair=expert_action["Res"])
			self.state["last_logp"] = ret_dict_c["c_logp"]
		
		
		ret_dict = {**ret_dict, **ret_dict_c}
		return ret_dict
