import torch as th
import torch.nn as nn
import numpy as np
from envs.res_unsat import ResUNSAT


class Encoder(nn.Module):
	def __init__(self, config):
		super(Encoder, self).__init__()

		self.rnn = nn.LSTM(input_size=config['emb_size'], hidden_size=config['emb_size'], num_layers=config['num_layers'],
						   batch_first=True, bidirectional=config['bidirectional'])

	def forward(self, embedded_inputs, input_lengths, hidden=None):
		# Pack padded batch of sequences for RNN module
		packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=True)
		# Forward pass through RNN
		outputs, hidden = self.rnn(packed, hidden)
		# Unpack padding
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
		# Return output and final hidden state
		return outputs, hidden

class Critic(nn.Module):
	def __init__(self, config):
		super(Critic, self).__init__()

		self.config = config
		self.net = nn.Sequential(
			nn.Linear(config['emb_size'], config['emb_size']),
			nn.Tanh(),
			nn.Linear(config['emb_size'], 1)
		)

	def forward(self, h_i):
		return self.net(h_i).squeeze(-1)


class AssignDecoder(nn.Module):
	def __init__(self, config):
		super(AssignDecoder, self).__init__()

		self.config = config
		self.env: ResUNSAT = config['env']
		q_width = config['emb_size']
		self.net = nn.Sequential(
			nn.Linear(q_width, config['emb_size']),
			nn.Tanh(),
			nn.Linear(config['emb_size'], 1),
			# Squeeze last 1 dim
			nn.Flatten(start_dim=-2)
		)

	def precheck_VA(self, A_pred):
		VA_pred = th.round(th.sigmoid(A_pred[0]))
		if A_pred[1]: # Invert GT
			VA_pred = 1 - VA_pred
		is_sat = self.env.is_satisfying([VA_pred])
		if is_sat:
			self.env.found_sat_VA = True
		VA_loss = nn.functional.binary_cross_entropy_with_logits(A_pred[0], VA_pred)
		return is_sat, VA_loss
		

	def min_VA_loss(self, A_pred):
		VA_set = self.env.formula.certificate
		losses = []
		for A_gt in VA_set:
			VA = th.tensor(A_gt).float().unsqueeze(0).to(self.config['device'])
			if A_pred[1]: # Invert GT
				VA = 1 - VA
			VA_loss = nn.functional.binary_cross_entropy_with_logits(A_pred[0], VA)
			losses.append(VA_loss)
		# Return min loss
		loss = th.stack(losses, dim=0).min()
		return loss
	
	def sat_loss(self, A_preds):
		losses = []
		for A_pred_i in A_preds:
			is_sat, VA_loss = self.precheck_VA(A_pred_i)
			if is_sat:
				losses.append(VA_loss)
			else:			
				loss_i = self.min_VA_loss(A_pred_i)
				losses.append(loss_i)
		# Return min loss
		if self.config['dual_loss_mode'] == "min":
			loss = th.stack(losses, dim=0).min().unsqueeze(0)
		elif self.config['dual_loss_mode'] == "sum":
			loss = th.stack(losses, dim=0).sum().unsqueeze(0)
		else:
			loss = th.stack(losses, dim=0).mean().unsqueeze(0)
		return loss

	def forward(self, state, A_gt):
		
		L_emb = state["literal_emb"]
		# Condition embeddings on h_i
		n_vars = L_emb.shape[1]//2
		L_emb = L_emb.reshape(1, 2, n_vars, self.config['emb_size'])

		if self.config['L_aggregate'] == "avg_in":
			V_emb = th.mean(L_emb, dim=1)
			A_preds = [(self.net(V_emb), False)]
		elif self.config['L_aggregate'] == "p_only":
			L_emb_p = L_emb[:, 0]
			A_preds = [(self.net(L_emb_p), False)]
			if self.env.mode != "train":
				L_emb_n = L_emb[:, 1]
				A_pred_n = self.net(L_emb_n)
				A_preds.append((A_pred_n, True))

		elif self.config['L_aggregate'] == "dual":
			# Produce two VAs: one from p and one from n
			L_emb_p, L_emb_n = L_emb[:, 0], L_emb[:, 1]
			A_pred_p = self.net(L_emb_p)
			A_pred_n = self.net(L_emb_n)
			# invert GT for negative
			A_preds = [(A_pred_p, False), (A_pred_n, True)]

		if A_gt is not None:
			sat_loss = self.sat_loss(A_preds)
		else:
			sat_loss = None
		# Get the discrete assignment
		for i in range(len(A_preds)):
			invert = A_preds[i][1]
			A_preds[i] = th.sigmoid(A_preds[i][0])
			A_preds[i] = th.round(A_preds[i])
			if invert: # Assume second is negative
				A_preds[i] = 1 - A_preds[i]
			
		ret_dict = {
			"A_pred": A_preds,
		}
		if sat_loss is not None:
			if self.config['res_sat_loss']:
				last_logp = state["last_logp"]
				# if last_logp is None: last_logp = th.Tensor([0.0]).to(self.config['device'])
				if last_logp is None: last_logp = 1.0
				# res_fact = th.exp(last_logp)
				res_fact = last_logp
				sat_loss = sat_loss * res_fact
			ret_dict["sat_loss"] = sat_loss
		return ret_dict

	# Batched API ---------------------------------------------------------
	def _bce_per_row(self, A_logit, VA_per_row, var_mask):
		"""A_logit: (B, max_vars). VA_per_row: (B, max_vars) float in {0,1}.
		var_mask: (B, max_vars) bool. Returns (B,) loss averaged per row over
		valid vars only."""
		bce = nn.functional.binary_cross_entropy_with_logits(
			A_logit, VA_per_row, reduction='none'
		)
		mask = var_mask.float()
		denom = mask.sum(dim=-1).clamp(min=1.0)
		return (bce * mask).sum(dim=-1) / denom

	def _min_loss_over_certs_batch(self, A_logit, A_gt_per_row, var_mask, invert):
		"""Per row, minimize BCE across the row's certificate set. Returns (B,)."""
		B, max_vars = A_logit.shape
		device = A_logit.device
		out = th.zeros(B, device=device)
		for b in range(B):
			cert_set = A_gt_per_row[b]
			if cert_set is None or len(cert_set) == 0:
				continue
			row_logit = A_logit[b:b+1]                              # (1, max_vars)
			row_mask = var_mask[b:b+1]
			losses_b = []
			for VA in cert_set:
				v_arr = np.asarray(VA, dtype=np.float32)
				# Pad assignment to max_vars; padded positions are masked out.
				pad = np.zeros(max_vars - len(v_arr), dtype=np.float32)
				v_full = np.concatenate([v_arr, pad])
				if invert:
					v_full = 1.0 - v_full
				v_t = th.from_numpy(v_full).to(device).unsqueeze(0)
				losses_b.append(self._bce_per_row(row_logit, v_t, row_mask).squeeze(0))
			out[b] = th.stack(losses_b, dim=0).min()
		return out

	def _precheck_VA_batch(self, A_pred_logit, var_mask, invert, running_mask):
		"""For each running row, check whether the rounded VA satisfies the
		formula. If yes, set env.found_sat_VA_b[b]=True and use that VA as the
		BCE target for the per-row loss. Returns (B,) loss + (B, max_vars) pred."""
		B, max_vars = A_pred_logit.shape
		device = A_pred_logit.device
		VA_pred = th.round(th.sigmoid(A_pred_logit))
		if invert:
			VA_pred_eff = 1.0 - VA_pred
		else:
			VA_pred_eff = VA_pred
		# Default loss = BCE(pred, pred) = ~zero (so it doesn't contribute much
		# unless precheck found SAT — matches original semantics).
		loss = self._bce_per_row(A_pred_logit, VA_pred_eff, var_mask)
		is_sat_b = th.zeros(B, dtype=th.bool, device=device)
		env = self.env
		va_pred_cpu = VA_pred_eff.detach().cpu().numpy()
		for b in range(B):
			if running_mask is not None and not bool(running_mask[b]):
				continue
			# Trim to row's real n_vars before passing to is_satisfying.
			n_vars_b = env.formulas_b[b].n_vars
			va_b = va_pred_cpu[b, :n_vars_b]
			sat = env._is_satisfying_row(b, [va_b])
			if sat:
				env.found_sat_VA_b[b] = True
				is_sat_b[b] = True
		return loss, VA_pred, is_sat_b

	def forward_batch(self, state, A_gt_per_row, running_mask=None):
		"""Batched forward. state needs:
		  literal_emb: (B, max_lits, emb)
		  var_mask:    (B, max_vars) bool
		  last_logp:   (B,) or None  (for res_sat_loss)
		A_gt_per_row: list[set | None] of length B (each row's certificate set)
		   or None when no supervised target is available.
		Returns dict with sat_loss: (B,) and A_pred: list[(B, max_vars)] tensors."""
		L_emb = state["literal_emb"]                              # (B, max_lits, emb)
		var_mask = state["var_mask"]                              # (B, max_vars)
		B, max_lits, hidden = L_emb.shape
		max_vars = max_lits // 2
		L_emb_r = L_emb.reshape(B, 2, max_vars, hidden)

		if self.config['L_aggregate'] == "avg_in":
			V_emb = th.mean(L_emb_r, dim=1)                       # (B, max_vars, emb)
			A_preds = [(self.net(V_emb), False)]
		elif self.config['L_aggregate'] == "p_only":
			A_preds = [(self.net(L_emb_r[:, 0]), False)]
			if self.env.mode != "train":
				A_preds.append((self.net(L_emb_r[:, 1]), True))
		elif self.config['L_aggregate'] == "dual":
			A_preds = [
				(self.net(L_emb_r[:, 0]), False),
				(self.net(L_emb_r[:, 1]), True),
			]
		else:
			raise ValueError(f"Unknown L_aggregate: {self.config['L_aggregate']}")

		sat_loss = None
		if A_gt_per_row is not None:
			# Per A_pred: precheck for SAT, else min-BCE over the certificate set.
			per_pred_losses = []
			for (A_logit, invert) in A_preds:
				pre_loss, _, is_sat_pre = self._precheck_VA_batch(
					A_logit, var_mask, invert, running_mask
				)
				min_loss = self._min_loss_over_certs_batch(
					A_logit, A_gt_per_row, var_mask, invert
				)
				# Use precheck loss for rows that already SAT, else min-cert loss.
				combined = th.where(is_sat_pre, pre_loss, min_loss)
				per_pred_losses.append(combined)
			stacked = th.stack(per_pred_losses, dim=0)              # (P, B)
			mode = self.config['dual_loss_mode']
			if mode == "min":
				sat_loss = stacked.min(dim=0).values
			elif mode == "sum":
				sat_loss = stacked.sum(dim=0)
			else:
				sat_loss = stacked.mean(dim=0)

			if self.config['res_sat_loss']:
				last_logp = state.get("last_logp")
				if last_logp is None:
					last_logp = 1.0
				sat_loss = sat_loss * last_logp

		# Discretized assignments to ship back as actions.
		A_pred_out = []
		for (A_logit, invert) in A_preds:
			va = th.round(th.sigmoid(A_logit))
			if invert:
				va = 1.0 - va
			A_pred_out.append(va)

		ret = {"A_pred": A_pred_out}
		if sat_loss is not None:
			ret["sat_loss"] = sat_loss                             # (B,)
		return ret