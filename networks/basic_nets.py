import torch as th
import torch.nn as nn
from envs.res_unsat import ResUNSAT


class Encoder(nn.Module):
	def __init__(self, opt):
		super(Encoder, self).__init__()

		self.rnn = nn.LSTM(input_size=opt.emb_size, hidden_size=opt.hidden_size, num_layers=opt.num_layers,
						   batch_first=True, bidirectional=opt.bidirectional)

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
	def __init__(self, opt):
		super(Critic, self).__init__()

		self.opt = opt
		self.net = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.Tanh(),
			nn.Linear(opt.hidden_size, 1)
		)

	def forward(self, h_i):
		return self.net(h_i).squeeze(-1)


class AssignDecoder(nn.Module):
	def __init__(self, opt):
		super(AssignDecoder, self).__init__()

		self.opt = opt
		self.env: ResUNSAT = opt.env
		q_width = opt.hidden_size
		self.net = nn.Sequential(
			nn.Linear(q_width, opt.hidden_size),
			nn.Tanh(),
			nn.Linear(opt.hidden_size, 1),
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
			VA = th.tensor(A_gt).float().unsqueeze(0).to(self.opt.device)
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
		if self.opt.dual_loss_mode == "min":
			loss = th.stack(losses, dim=0).min().unsqueeze(0)
		elif self.opt.dual_loss_mode == "sum":
			loss = th.stack(losses, dim=0).sum().unsqueeze(0)
		else:
			loss = th.stack(losses, dim=0).mean().unsqueeze(0)
		return loss

	def forward(self, state, A_gt):
		
		L_emb = state["literal_emb"]
		# Condition embeddings on h_i
		n_vars = L_emb.shape[1]//2
		L_emb = L_emb.reshape(1, 2, n_vars, self.opt.hidden_size)

		if self.opt.L_aggregate == "avg_in":
			V_emb = th.mean(L_emb, dim=1)
			A_preds = [(self.net(V_emb), False)]
		elif self.opt.L_aggregate == "p_only":
			L_emb_p = L_emb[:, 0]
			A_preds = [(self.net(L_emb_p), False)]
			if self.env.mode != "train":
				L_emb_n = L_emb[:, 1]
				A_pred_n = self.net(L_emb_n)
				A_preds.append((A_pred_n, True))

		elif self.opt.L_aggregate == "dual":
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
			if self.opt.res_sat_loss:
				last_logp = state["last_logp"]
				# if last_logp is None: last_logp = th.Tensor([0.0]).to(self.opt.device)
				if last_logp is None: last_logp = 1.0
				# res_fact = th.exp(last_logp)
				res_fact = last_logp
				sat_loss = sat_loss * res_fact
			ret_dict["sat_loss"] = sat_loss
		return ret_dict