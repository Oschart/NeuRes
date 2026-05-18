import torch as th
import torch.nn as nn
import numpy as np
from common.problem import Formula
from common.batched_formula import BatchedFormula

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_size = config['emb_size']
        if config['n_rounds'].isnumeric():
            self.n_rounds = int(config['n_rounds'])
        else:
           self.n_rounds = config['n_rounds']

        self.device = config['device']

        self.init_ts = th.ones(1).to(self.device)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, config['emb_size'])
        self.C_init = nn.Linear(1, config['emb_size'])

        self.L_msg = MLP(self.emb_size, self.emb_size, self.emb_size)
        self.C_msg = MLP(self.emb_size, self.emb_size, self.emb_size)

        self.L_update = nn.LSTM(self.emb_size*2, self.emb_size)
        self.L_norm   = nn.LayerNorm(self.emb_size)
        self.C_update = nn.LSTM(self.emb_size, self.emb_size)
        self.C_norm   = nn.LayerNorm(self.emb_size)
        self.alpha = config['partial_round_alpha']

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

    # ---------------------------------------------------------------
    # Batched API. State shapes:
    #   L_state[h|c]: (1, B, max_lits, emb)
    #   C_state[h|c]: (1, B, H_max,    emb)
    #   L_unpack:     (B, max_lits, H_max)  — owned, mutated as clauses grow
    # All padded positions are kept zero between rounds via _zero_padded_states.
    # ---------------------------------------------------------------
    def init_batch(self, batched: BatchedFormula) -> dict:
        self.batched = batched
        B = batched.B
        self.B_size = B
        self.max_vars_b = batched.max_vars
        self.max_lits_b = batched.max_lits
        self.H_max_b = batched.H_max

        # The batched-formula owns the initial L_unpack / clause_mask tensors.
        # We take them as live references — embed_clause_batch mutates them in
        # place as new clauses are added. This keeps every consumer (selectors,
        # decoder) reading from the same underlying buffer.
        self.L_unpack_b = batched.L_unpack
        self.lit_mask_b = batched.lit_mask
        self.clause_mask_b = batched.clause_mask
        self.flip_idx_b = batched.flip_idx
        # CPU-side mirror of per-row clause counts. Maintained in lockstep with
        # the GPU tensor below so consumers (current_C tracking, slot lookups)
        # never need a `.item()` sync.
        self.current_clause_count_cpu = batched.n_clauses.detach().cpu().numpy().astype(np.int64).copy()
        self.current_clause_count = th.from_numpy(self.current_clause_count_cpu.copy()).to(self.device)
        self.current_C_int = int(self.current_clause_count_cpu.max())

        init_ts = self.init_ts.to(self.device)
        L_init = self.L_init(init_ts).view(1, 1, 1, -1)  # (1,1,1,emb)
        C_init = self.C_init(init_ts).view(1, 1, 1, -1)
        L_h = L_init.repeat(1, B, self.max_lits_b, 1)
        L_c = th.zeros(1, B, self.max_lits_b, self.emb_size, device=self.device)
        # C_state is grown dynamically alongside current_C. Initial width is
        # max_b(initial_n_clauses) — we never compute on slots beyond
        # current_clause_count[b], and embed_clause_batch extends the buffer
        # by cat when current_C grows (typically +1 column per step).
        C_h = C_init.repeat(1, B, self.current_C_int, 1)
        C_c = th.zeros(1, B, self.current_C_int, self.emb_size, device=self.device)
        self.L_state_b = (L_h, L_c)
        self.C_state_b = (C_h, C_c)
        self._zero_padded_states_b()

        # Determine n_rounds — for V/C modes, take the per-batch max so every
        # row gets at least as many rounds as it would have at B=1.
        if self.n_rounds in ["V", "v"]:
            n_rounds = int(batched.n_vars.max().item()) + 1
        elif self.n_rounds in ["C", "c"]:
            n_rounds = int(batched.n_clauses.max().item())
        else:
            n_rounds = self.n_rounds

        for _ in range(n_rounds):
            self.perform_full_round_batch()

        return {
            "clauses": self.C_state_b[0].squeeze(0),    # (B, H_max, emb)
            "literals": self.L_state_b[0].squeeze(0),   # (B, max_lits, emb)
        }

    def _zero_padded_states_b(self):
        """Zero hidden states at padded literal/clause positions so subsequent
        rounds and downstream readers (selectors, decoder) see clean zeros.
        clause_mask_b is sized H_max (the static buffer); we slice to the
        embedder's current C_state width since C_state grows dynamically."""
        C = self.C_state_b[0].size(2)
        lit_keep = self.lit_mask_b.unsqueeze(0).unsqueeze(-1).float()                  # (1, B, max_lits, 1)
        clause_keep = self.clause_mask_b[:, :C].unsqueeze(0).unsqueeze(-1).float()     # (1, B, C, 1)
        self.L_state_b = (self.L_state_b[0] * lit_keep, self.L_state_b[1] * lit_keep)
        self.C_state_b = (self.C_state_b[0] * clause_keep, self.C_state_b[1] * clause_keep)

    def flip_lit_emb(self, L: th.Tensor) -> th.Tensor:
        """L: (B, max_lits, emb). Per-row swap pos↔neg literals via gather."""
        return th.gather(L, dim=1, index=self.flip_idx_b.unsqueeze(-1).expand_as(L))

    def perform_full_round_batch(self):
        # Operate on the current effective clause width — L_unpack and C_state
        # past current_C are unoccupied padding (kept zero) so we can skip
        # their LSTM/bmm work entirely.
        C = self.C_state_b[0].size(2)

        # Literal → clause direction.
        L_hidden = self.L_state_b[0].squeeze(0)                # (B, max_lits, emb)
        L_pre_msg = self.L_msg(L_hidden)                        # (B, max_lits, emb)
        # Slice L_unpack to current C, then clone so subsequent in-place writes
        # in embed_clause_batch don't break autograd's version check on the
        # tensor saved here.
        LU = self.L_unpack_b[:, :, :C].detach().clone()         # (B, max_lits, C)
        LC_msg = th.bmm(LU.transpose(-1, -2), L_pre_msg)        # (B, C, emb)

        BC = self.B_size * C
        LC_flat = LC_msg.reshape(1, BC, self.emb_size)
        C_h_flat = self.C_state_b[0].reshape(1, BC, self.emb_size)
        C_c_flat = self.C_state_b[1].reshape(1, BC, self.emb_size)
        _, (C_h_new, C_c_new) = self.C_update(LC_flat, (C_h_flat, C_c_flat))
        C_h_new = C_h_new.reshape(1, self.B_size, C, self.emb_size)
        C_c_new = C_c_new.reshape(1, self.B_size, C, self.emb_size)
        self.C_state_b = (self.C_norm(C_h_new), C_c_new)

        # Clause → literal direction.
        C_hidden = self.C_state_b[0].squeeze(0)                # (B, C, emb)
        C_pre_msg = self.C_msg(C_hidden)
        CL_msg = th.bmm(LU, C_pre_msg)                          # (B, max_lits, emb)
        L_flipped = self.flip_lit_emb(self.L_state_b[0].squeeze(0))  # (B, max_lits, emb)
        L_input = th.cat([CL_msg, L_flipped], dim=-1)           # (B, max_lits, 2*emb)

        BL = self.B_size * self.max_lits_b
        L_input_flat = L_input.reshape(1, BL, 2 * self.emb_size)
        L_h_flat = self.L_state_b[0].reshape(1, BL, self.emb_size)
        L_c_flat = self.L_state_b[1].reshape(1, BL, self.emb_size)
        _, (L_h_new, L_c_new) = self.L_update(L_input_flat, (L_h_flat, L_c_flat))
        L_h_new = L_h_new.reshape(1, self.B_size, self.max_lits_b, self.emb_size)
        L_c_new = L_c_new.reshape(1, self.B_size, self.max_lits_b, self.emb_size)
        self.L_state_b = (self.L_norm(L_h_new), L_c_new)

        self._zero_padded_states_b()

    def perform_partial_round_batch(self, new_slot_per_row: th.LongTensor, has_real_clause: th.BoolTensor):
        """Same as perform_full_round_batch but blends new state with old via
        per-row alpha. Rows whose `has_real_clause[b]` is True get alpha=1 at
        their new slot (no blend); other slots/rows blend at self.alpha."""
        C = self.C_state_b[0].size(2)
        L_hidden = self.L_state_b[0].squeeze(0)
        L_pre_msg = self.L_msg(L_hidden)
        LU = self.L_unpack_b[:, :, :C].detach().clone()
        LC_msg = th.bmm(LU.transpose(-1, -2), L_pre_msg)

        BC = self.B_size * C
        LC_flat = LC_msg.reshape(1, BC, self.emb_size)
        C_h_flat = self.C_state_b[0].reshape(1, BC, self.emb_size)
        C_c_flat = self.C_state_b[1].reshape(1, BC, self.emb_size)
        _, (C_h_new, C_c_new) = self.C_update(LC_flat, (C_h_flat, C_c_flat))
        C_h_new = C_h_new.reshape(1, self.B_size, C, self.emb_size)
        C_c_new = C_c_new.reshape(1, self.B_size, C, self.emb_size)

        C_hidden = self.C_state_b[0].squeeze(0)
        C_pre_msg = self.C_msg(C_hidden)
        CL_msg = th.bmm(LU, C_pre_msg)
        L_flipped = self.flip_lit_emb(self.L_state_b[0].squeeze(0))
        L_input = th.cat([CL_msg, L_flipped], dim=-1)

        BL = self.B_size * self.max_lits_b
        L_input_flat = L_input.reshape(1, BL, 2 * self.emb_size)
        L_h_flat = self.L_state_b[0].reshape(1, BL, self.emb_size)
        L_c_flat = self.L_state_b[1].reshape(1, BL, self.emb_size)
        _, (L_h_new, L_c_new) = self.L_update(L_input_flat, (L_h_flat, L_c_flat))
        L_h_new = L_h_new.reshape(1, self.B_size, self.max_lits_b, self.emb_size)
        L_c_new = L_c_new.reshape(1, self.B_size, self.max_lits_b, self.emb_size)

        # Per-row alpha for clauses: alpha everywhere, except 1 at the new slot
        # of rows that derived a real clause. Sized to current_C.
        alpha_c = th.full((self.B_size, C), self.alpha, device=self.device)
        if has_real_clause.any():
            rows = th.nonzero(has_real_clause, as_tuple=False).flatten()
            alpha_c[rows, new_slot_per_row[rows]] = 1.0
        alpha_c = alpha_c.unsqueeze(0).unsqueeze(-1)  # (1, B, C, 1)
        alpha_l = th.full((1, self.B_size, self.max_lits_b, 1), self.alpha, device=self.device)

        C_h_blend = (1 - alpha_c) * self.C_state_b[0] + alpha_c * C_h_new
        C_c_blend = (1 - alpha_c) * self.C_state_b[1] + alpha_c * C_c_new
        self.C_state_b = (C_h_blend, C_c_blend)

        L_h_blend = (1 - alpha_l) * self.L_state_b[0] + alpha_l * L_h_new
        L_c_blend = (1 - alpha_l) * self.L_state_b[1] + alpha_l * L_c_new
        self.L_state_b = (L_h_blend, L_c_blend)

        self._zero_padded_states_b()

    def embed_clause_batch(self, new_clauses_per_row, running_np: np.ndarray) -> dict:
        """Per row: if a real resolvent was derived (running_np[b] True AND
        new_clauses_per_row[b] non-empty), claim slot current_clause_count[b],
        write its literal indicators into L_unpack, flip clause_mask True at
        that slot, then advance the counter. Rows that failed to derive a
        clause (e.g. unsupervised picks of non-resolvable / duplicate pairs)
        do NOT claim a slot — the counter stays put, so the embedder's slot
        index keeps matching env.clauses_b index in subsequent steps. The
        failure path then degenerates to a pure MP round on the unchanged-
        shape state, matching the original embed_clause([]) behavior.

        running_np is a CPU numpy bool array (typically `~env.done_b`) — we
        avoid round-tripping a GPU tensor here so the loop stays sync-free."""
        B = self.B_size
        device = self.device

        # ---- Phase 1: CPU-side index gathering (no GPU syncs) -------------
        # Walk per-row Python clause lists and collect flat (row, lit, slot)
        # triples for a single batched scatter. new_slots_cpu[b] holds the
        # slot claimed by row b this step; meaningful only when has_real_cpu[b].
        rows_lit, lit_pos, slot_pos = [], [], []
        new_slots_cpu = np.zeros(B, dtype=np.int64)
        has_real_cpu = np.zeros(B, dtype=bool)
        for b in range(B):
            if not running_np[b]:
                continue
            new_cs = new_clauses_per_row[b] if new_clauses_per_row[b] else []
            if not new_cs:
                # Pure MP-round step for this row: no slot claimed so the
                # embedder slot index stays aligned with env.clauses_b index.
                continue
            slot = int(self.current_clause_count_cpu[b])
            if slot >= self.H_max_b:
                continue
            new_slots_cpu[b] = slot
            c_new = new_cs[0]
            for lit in c_new:
                var = abs(lit) - 1
                rows_lit.append(b)
                slot_pos.append(slot)
                lit_pos.append(var if lit > 0 else self.max_vars_b + var)
            has_real_cpu[b] = True
            self.current_clause_count_cpu[b] += 1

        # ---- Phase 2: single batched GPU writes ---------------------------
        # current_C_int now reflects the post-step max. Track old size so we
        # know whether C_state needs to grow.
        old_C = self.C_state_b[0].size(2)
        self.current_C_int = int(self.current_clause_count_cpu.max())
        self.current_clause_count = th.from_numpy(self.current_clause_count_cpu.copy()).to(device)
        new_slot_per_row = th.from_numpy(new_slots_cpu).to(device)
        has_real = th.from_numpy(has_real_cpu.copy()).to(device)

        with th.no_grad():
            if rows_lit:
                idx_b = th.as_tensor(rows_lit, dtype=th.long, device=device)
                idx_l = th.as_tensor(lit_pos, dtype=th.long, device=device)
                idx_s = th.as_tensor(slot_pos, dtype=th.long, device=device)
                ones = th.ones(len(rows_lit), device=device, dtype=self.L_unpack_b.dtype)
                self.L_unpack_b.index_put_((idx_b, idx_l, idx_s), ones)
            if has_real_cpu.any():
                real_rows_np = np.where(has_real_cpu)[0].astype(np.int64)
                real_rows = th.from_numpy(real_rows_np).to(device)
                self.clause_mask_b[real_rows, new_slot_per_row[real_rows]] = True

        # Grow C_state only if some row actually claimed a new column.
        # Padded positions stay zero (and clause_mask_b is False there), so
        # _zero_padded_states_b keeps them inert in subsequent rounds.
        new_C = self.current_C_int
        if new_C > old_C:
            delta = new_C - old_C
            pad_h = th.zeros(1, self.B_size, delta, self.emb_size, device=device, dtype=self.C_state_b[0].dtype)
            pad_c = th.zeros(1, self.B_size, delta, self.emb_size, device=device, dtype=self.C_state_b[1].dtype)
            self.C_state_b = (
                th.cat([self.C_state_b[0], pad_h], dim=2),
                th.cat([self.C_state_b[1], pad_c], dim=2),
            )

        # Re-init the LSTM cell at the newly-claimed slot of each row that
        # derived a real clause. Use th.where to build new C_state tensors
        # rather than mutating in place — C_state_b[0] is the output of the
        # previous round's LSTM and still part of the autograd graph.
        if has_real_cpu.any():
            init_ts = self.init_ts.to(self.device)
            C_init_h = self.C_init(init_ts).view(1, 1, 1, -1)          # (1,1,1,emb)
            slot_mask = th.zeros(B, new_C, dtype=th.bool, device=device)
            real_rows_np = np.where(has_real_cpu)[0].astype(np.int64)
            rows = th.from_numpy(real_rows_np).to(device)
            slot_mask[rows, new_slot_per_row[rows]] = True
            slot_mask4 = slot_mask.unsqueeze(0).unsqueeze(-1)           # (1, B, new_C, 1)
            h_target = C_init_h.expand_as(self.C_state_b[0])
            c_target = th.zeros_like(self.C_state_b[1])
            self.C_state_b = (
                th.where(slot_mask4, h_target, self.C_state_b[0]),
                th.where(slot_mask4, c_target, self.C_state_b[1]),
            )

        if self.alpha != 1:
            self.perform_partial_round_batch(new_slot_per_row, has_real)
        else:
            for _ in range(self.config['mp_per_res']):
                self.perform_full_round_batch()

        # Per-row gather of the embedding at the newly added slot.
        slot_idx = new_slot_per_row.view(B, 1, 1).expand(B, 1, self.emb_size)
        new_clause_emb = th.gather(self.C_state_b[0].squeeze(0), 1, slot_idx).squeeze(1)  # (B, emb)
        return {
            "new_clause_emb": new_clause_emb,
            "new_slot_per_row": new_slot_per_row,
            "has_real_clause": has_real,
            "clauses": self.C_state_b[0].squeeze(0),
            "literals": self.L_state_b[0].squeeze(0),
        }

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
            for i in range(self.config['mp_per_res']):
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

