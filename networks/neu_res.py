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
	def __init__(self, config):
		super(NeuRes, self).__init__()

		self.config = config
		self.env: ResUNSAT = config['env']
		# Embedding dimension
		self.emb_size = config['emb_size']
		# (Decoder) hidden size
		self.hidden_size = config['hidden_size']
		# Bidirectional Encoder
		self.bidirectional = config['bidirectional']
		self.num_directions = 2 if config['bidirectional'] else 1
		self.enc_dec_cycle = config['enc_dec_cycle']
		self.num_layers = config['num_layers']
		self.eps_greed = 0.0
		self.device = config['device']
		self.reset()

		self.embedder = MP_Embedder(config)
		self.c_selector = make_attn_module(config['c_selector'], config)

		if config['predict_sat']:
			self.sat_vote = MLP(config['emb_size'], config['emb_size'], 1)
			self.vote_bias = nn.Parameter(th.zeros(1))


		if config['critic']:
			self.critic = Critic(config)
		else:
			self.critic = zero_net()

		if "sat" in config['dataset'] or "OmNeuRes" in config['solver']:
			self.assign_decoder = AssignDecoder(config)
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
		
		if self.config['attn_QK'] == "Emb_reuse":
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
		if self.config['predict_sat']:
			L_emb = self.state["literal_emb"]
			# n_vars = L_emb.shape[1]//2
			sat_votes = self.sat_vote(L_emb)
			ret_dict["sat_vote"] = th.mean(sat_votes) + self.vote_bias

		is_sat_train = expert_action is not None and expert_action["VA"] is not None
		unsupervised = not self.env.supervised
		if self.assign_decoder is not None and (is_sat_train or unsupervised):
			a_dict = self.assign_decoder(self.state, expert_action["VA"])
			ret_dict = {**ret_dict, **a_dict}

		if self.is_sat and not self.config['res_sat_loss']:
			if self.config['no_res']:
				ret_dict_c = {}
			else:
				with th.no_grad():
					ret_dict_c = self.c_selector(self.state, expert_pair=expert_action["Res"])
		else:
			ret_dict_c = self.c_selector(self.state, expert_pair=expert_action["Res"])
			self.state["last_logp"] = ret_dict_c["c_logp"]


		ret_dict = {**ret_dict, **ret_dict_c}
		return ret_dict

	# ====================================================================
	# Batched API. The state dict here holds (B,...) tensors and per-row
	# Python collections (taken_set, taken_map). Done rows still flow through
	# the forward; the solver masks their losses via running_mask in ep_data.
	# ====================================================================
	def reset_batch(self):
		self.state = {
			"clause_emb": None,
			"literal_emb": None,
			"clause_mask": None,
			"lit_mask": None,
			"var_mask": None,
			"flip_idx": None,
			"res_mask_pair": None,
			"taken_set": [],
			"taken_map": [],
			"running_mask": None,
			"last_logp": None,
		}

	def setup_batch(self, formulas, sample_idxs=None):
		self.reset_batch()
		self.formulas_b = list(formulas)
		self.is_sat_b = th.tensor([f.is_sat for f in formulas], dtype=th.bool, device=self.device)
		# Env builds the BatchedFormula and per-row state.
		batched = self.env.init_batch(formulas, sample_idxs=sample_idxs)
		self.embedder.init_batch(batched)
		B = len(formulas)
		self.state["taken_set"] = [set() for _ in range(B)]
		self.state["taken_map"] = [{} for _ in range(B)]
		self.state["lit_mask"] = batched.lit_mask
		self.state["var_mask"] = batched.var_mask
		self.state["flip_idx"] = batched.flip_idx
		self._refresh_state_from_embedder()

	def _refresh_state_from_embedder(self):
		"""Pull current clause_emb / literal_emb / clause_mask references from
		the embedder. clause_emb / clause_mask cover up to the embedder's
		current_clause_count.max() — selectors slice further with [:, :C, :]."""
		self.state["clause_emb"] = self.embedder.C_state_b[0].squeeze(0)   # (B, H_max, emb)
		self.state["literal_emb"] = self.embedder.L_state_b[0].squeeze(0)  # (B, max_lits, emb)
		self.state["clause_mask"] = self.embedder.clause_mask_b              # (B, H_max)

	def _current_C(self) -> int:
		"""Effective clause-grid width = max over running rows of current_clause_count.
		Selectors operate on (B, current_C, ...) slices of the embedder's H_max-sized
		buffers, which keeps padding overhead bounded by the slowest row.
		The embedder maintains a CPU-side mirror so this is a pure Python read."""
		return self.embedder.current_C_int

	def perform_mp_round_batch(self):
		self.embedder.perform_full_round_batch()
		self._refresh_state_from_embedder()

	def add_new_clauses_batched(self, new_clauses_per_row, running_mask):
		# running_mask (GPU tensor) is kept in the signature for caller symmetry
		# with forward_batch, but we read straight from env.done_b to avoid a
		# device→host sync on the per-step path.
		running_np = ~self.env.done_b
		ret = self.embedder.embed_clause_batch(new_clauses_per_row, running_np)
		self._refresh_state_from_embedder()
		return ret

	def forward_batch(self, expert_actions, running_mask):
		"""expert_actions: list[dict] of length B (each {"Res": ..., "VA": ...}) or None
		when running unsupervised. running_mask: (B,) bool tensor on device."""
		self._refresh_state_from_embedder()
		self.state["running_mask"] = running_mask

		# Slice clause_emb / clause_mask to the current effective width so the
		# attention grids don't waste compute on positions no row has reached.
		C = self._current_C()
		# Defensive: a degenerate batch where every row has zero clauses is
		# impossible here (init_batch always sets ≥1 clause per row).
		clause_emb_full = self.state["clause_emb"]
		clause_mask_full = self.state["clause_mask"]
		self.state["clause_emb"] = clause_emb_full[:, :C, :]
		self.state["clause_mask"] = clause_mask_full[:, :C]

		# res_mask_pair comes from the env's per-row res_mask_b. Pad to (B, C, C).
		self.state["res_mask_pair"] = self.env.res_mask_batched_t(C)

		ret_dict = {}
		if self.config['predict_sat']:
			L_emb = self.state["literal_emb"]
			lit_mask = self.state["lit_mask"].float().unsqueeze(-1)            # (B, max_lits, 1)
			sat_votes = self.sat_vote(L_emb)                                   # (B, max_lits, 1)
			# Per-row mean over real literals.
			denom = lit_mask.sum(dim=1).clamp(min=1.0)
			ret_dict["sat_vote"] = (sat_votes * lit_mask).sum(dim=1) / denom + self.vote_bias  # (B, 1)
			ret_dict["sat_vote"] = ret_dict["sat_vote"].squeeze(-1)            # (B,)

		# AssignDecoder branch — runs when SAT supervision is available or
		# we're unsupervised (so the model can also propose VAs at eval).
		expert_VA_per_row = None
		if expert_actions is not None:
			expert_VA_per_row = [a.get("VA") for a in expert_actions]
		any_VA_supervised = expert_VA_per_row is not None and any(va is not None for va in expert_VA_per_row)
		unsupervised = not self.env.supervised
		if self.assign_decoder is not None and (any_VA_supervised or unsupervised):
			# Per-row certificate set (formula.certificate is the GT VA set for SAT).
			A_gt_per_row = [
				(self.formulas_b[b].certificate if self.is_sat_b[b] else None)
				for b in range(len(self.formulas_b))
			]
			a_dict = self.assign_decoder.forward_batch(self.state, A_gt_per_row, running_mask=running_mask)
			ret_dict = {**ret_dict, **a_dict}

		# Resolution selector. Expert pairs are per-row "Res" entries.
		expert_pairs = None
		if expert_actions is not None:
			pairs = [a.get("Res") for a in expert_actions]
			# An all-None list means "no supervision this step" (e.g.
			# unsupervised validation / preroll where the env's guide yields
			# {"Res": None}). Pass None so the selector takes the argmax
			# branch instead of gathering at the (0,0) sentinel for every row.
			if any(p is not None for p in pairs):
				expert_pairs = pairs
		if self.config.get('no_res'):
			ret_dict_c = {}
		else:
			# When ALL rows are SAT and we're not training resolution on SAT,
			# match the original "no-grad selector" behavior.
			if self.is_sat_b.all() and not self.config['res_sat_loss']:
				with th.no_grad():
					ret_dict_c = self._call_selector_batch(expert_pairs)
			else:
				ret_dict_c = self._call_selector_batch(expert_pairs)
				self.state["last_logp"] = ret_dict_c["c_logp"]
		ret_dict = {**ret_dict, **ret_dict_c}

		# Restore full clause_emb/mask refs in state so the next round's
		# embedder.perform_full_round_batch sees the full buffer.
		self.state["clause_emb"] = clause_emb_full
		self.state["clause_mask"] = clause_mask_full
		return ret_dict

	def _call_selector_batch(self, expert_pairs):
		# All three selectors expose forward_batch with the same signature.
		return self.c_selector.forward_batch(self.state, expert_pairs=expert_pairs)
