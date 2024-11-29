import torch as th
import torch.nn as nn
import numpy as np
from common.problem import Formula

class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = th.relu(self.l1(x))
    x = th.relu(self.l2(x))
    x = self.l3(x)
    return x


class MP_Embedder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.emb_size = opt.emb_size
        if opt.n_rounds.isnumeric():
            self.n_rounds = int(opt.n_rounds)
        else:
           self.n_rounds = opt.n_rounds

        self.device = opt.device

        self.init_ts = th.ones(1).to(self.device)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, opt.emb_size)
        self.C_init = nn.Linear(1, opt.emb_size)

        self.L_msg = MLP(self.emb_size, self.emb_size, self.emb_size)
        self.C_msg = MLP(self.emb_size, self.emb_size, self.emb_size)

        self.L_update = nn.LSTM(self.emb_size*2, self.emb_size)
        self.L_norm   = nn.LayerNorm(self.emb_size)
        self.C_update = nn.LSTM(self.emb_size, self.emb_size)
        self.C_norm   = nn.LayerNorm(self.emb_size)
        self.alpha = opt.partial_round_alpha

        self.to(self.device)

    def reset(self):
        self.L_state = None
        self.C_state = None
        self.n_vars = None
        self.n_lits = None

    def init(self, formula: Formula):
        n_vars    = formula.n_vars
        n_lits    = formula.n_lits
        n_clauses = formula.n_clauses

        ts_L_unpack_indices = th.Tensor(formula.L_unpack_indices).t().long()
        
        init_ts = self.init_ts.to(self.device)
        # 1 x n_lits x dim & 1 x n_clauses x dim
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, th.zeros(1, n_lits, self.emb_size).to(self.device))
        C_state = (C_init, th.zeros(1, n_clauses, self.emb_size).to(self.device))
        L_unpack  = th.sparse.FloatTensor(ts_L_unpack_indices, th.ones(formula.n_cells), th.Size([n_lits, n_clauses])).to_dense().to(self.device)

        self.L_state = L_state
        self.C_state = C_state
        self.n_vars = n_vars
        self.n_lits = n_lits
        self.L_unpack = L_unpack

        if self.n_rounds in ["V", "v"]:
           n_rounds = n_vars + 1
        elif self.n_rounds in ["C", "c"]:
            n_rounds = n_clauses
        else:
            n_rounds = self.n_rounds

        for _ in range(n_rounds):
            self.perform_full_round()

        literal_embeddings = self.L_state[0].squeeze(0)
        clause_embeddings = self.C_state[0].squeeze(0)


        return {"clauses": clause_embeddings, "literals": literal_embeddings}


    def perform_full_round(self):
        # n_lits x dim
        L_hidden = self.L_state[0].squeeze(0)
        L_pre_msg = self.L_msg(L_hidden)
        # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
        LC_msg = th.matmul(self.L_unpack.t(), L_pre_msg)

        _, self.C_state = self.C_update(LC_msg.unsqueeze(0), self.C_state)
        self.C_state = (self.C_norm(self.C_state[0]), self.C_state[1])

        # n_clauses x dim
        C_hidden = self.C_state[0].squeeze(0)
        C_pre_msg = self.C_msg(C_hidden)
        # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
        CL_msg = th.matmul(self.L_unpack, C_pre_msg)

        _, self.L_state = self.L_update(th.cat([CL_msg, self.flip(self.L_state[0].squeeze(0), self.n_vars)], dim=1).unsqueeze(0), self.L_state)
        self.L_state = (self.L_norm(self.L_state[0]), self.L_state[1])

        return
    
    def perform_partial_round(self):
        new_idx = -1
        # Update all embeddings with an update weight factor to control the magnitude of the update
        # Except the new clause embedding, which is updated with a weight factor of 1
        # n_lits x dim
        L_hidden = self.L_state[0].squeeze(0)
        L_pre_msg = self.L_msg(L_hidden)
        # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
        LC_msg = th.matmul(self.L_unpack.t(), L_pre_msg)

        _, C_state_new = self.C_update(LC_msg.unsqueeze(0), self.C_state)

        # n_clauses x dim
        C_hidden = self.C_state[0].squeeze(0)
        C_pre_msg = self.C_msg(C_hidden)
        # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
        CL_msg = th.matmul(self.L_unpack, C_pre_msg)

        _, L_state_new = self.L_update(th.cat([CL_msg, self.flip(self.L_state[0].squeeze(0), self.n_vars)], dim=1).unsqueeze(0), self.L_state)

        # Construct alpha weight vector
        n_clauses = self.C_state[0].shape[1]
        alpha_vec_c = th.ones(n_clauses).to(self.device) * self.alpha
        alpha_vec_c[new_idx] = 1
        alpha_vec_c = alpha_vec_c.unsqueeze(0).unsqueeze(2)
        
        alpha_vec_l = th.ones(self.n_lits).to(self.device) * self.alpha
        alpha_vec_l = alpha_vec_l.unsqueeze(0).unsqueeze(2)
        

        # Update all other embeddings (inc. literals) with a finetune factor
        C_state_h = (1-alpha_vec_c) * self.C_state[0] + alpha_vec_c * C_state_new[0]
        C_state_c = (1-alpha_vec_c) * self.C_state[1] + alpha_vec_c * C_state_new[1]
        self.C_state = (C_state_h, C_state_c)

        L_state_h = (1-alpha_vec_l) * self.L_state[0] + alpha_vec_l * L_state_new[0]
        L_state_c = (1-alpha_vec_l) * self.L_state[1] + alpha_vec_l * L_state_new[1]       
        self.L_state = (L_state_h, L_state_c)
        
        return


    def flip(self, msg, n_vars):
        return th.cat([msg[n_vars:2*n_vars, :], msg[:n_vars, :]], dim=0)

    def embed_clause(self, clause):
        # clause: 1 x n_lits
        # Make single-clause formula
        if isinstance(clause, list):
            n_clauses = len(clause)
            c_problem = Formula(self.n_vars, clause, False, None)
        else:
            n_clauses = 1
            c_problem = Formula(self.n_vars, [clause], False, None)
        ts_L_unpack_indices = th.Tensor(c_problem.L_unpack_indices).t().long()
        # L_init = self.L_init(self.init_ts).view(1, 1, -1)
        C_init = self.C_init(self.init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)
        # L_state = (L_init, th.zeros(1, 1, self.emb_size).to(self.device))
        C_state = (C_init, th.zeros(1, n_clauses, self.emb_size).to(self.device))
        L_unpack  = th.sparse.FloatTensor(ts_L_unpack_indices, th.ones(c_problem.n_cells), th.Size([self.n_lits, n_clauses])).to_dense().to(self.device)

        
        # Extend the state
        self.C_state = (th.cat([self.C_state[0], C_state[0]], dim=1), th.cat([self.C_state[1], C_state[1]], dim=1))
        self.L_unpack = th.cat([self.L_unpack, L_unpack], dim=1)

        # Perform one round
        if self.alpha != 1:
            self.perform_partial_round()
        else:
            for i in range(self.opt.mp_per_res):
                self.perform_full_round()

        new_clause_emb = self.C_state[0][:, -1, :]
        literal_embeddings = self.L_state[0].squeeze(0)
        clause_embeddings = self.C_state[0].squeeze(0)

        ret = {
            "new_clause_emb": new_clause_emb, 
            "clauses": clause_embeddings, 
            "literals": literal_embeddings
        }

        return ret

