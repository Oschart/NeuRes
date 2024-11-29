from typing import Any, Dict, Optional, NamedTuple, Union
import torch as th
import numpy as np
import argparse
from common.problem import Formula

def constant(p):
    return 1


def linear(p):
    return 1-p


def middle_drop(p):
    eps = 0.75
    if 1-p < eps:
        return eps*0.1
    return 1-p


def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1-p < eps:
        return eps
    return 1-p


def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1-p < eps1:
        if 1-p < eps2:
            return eps2*0.5
        return eps1*0.1
    return 1-p


schedules = {
    'linear': linear,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):

    def __init__(self, v, schedule):
        self.n = 0.
        self.v = v
        self.schedule = schedules[schedule]

    def __call__(self, remaining_ratio):
        current_value = self.v*self.schedule(1 - remaining_ratio)
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps/self.nvalues)


class PolicySpec(NamedTuple):
    type: str
    embed_specs: Dict[str, Any]
    base_specs: Dict[str, Any]


class RolloutData(NamedTuple):
    log_probs: th.Tensor
    returns: th.Tensor

class Episode(NamedTuple):
    idx: int
    ret_dict: dict
    extras: dict


def compute_returns(rewards: np.array, gamma: float):
    """Compute discounted returns from immediate rewards."""
    t_steps = np.arange(rewards.size)
    r = rewards * gamma**t_steps
    r = r[::-1].cumsum()[::-1] / gamma**t_steps
    return r

def get_time_horizon(n, gamma: float):
    """Reverse geometric sequence with factor gamma."""
    t_steps = th.arange(n)
    # h = (gamma**t_steps)[::-1]
    h = (gamma**t_steps).flip(0)
    return h


def mask_grid(grid, pair_set, keep_mask: np.array = None):
    """log-softmax with only valid inputs, excluding zero padded parts"""
    N = grid.size(1)
    grid_exc = grid.clone()
    neg_inf = -np.inf
    for i, j in pair_set:
        grid_exc[:, i, j] = neg_inf
        grid_exc[:, j, i] = neg_inf
    # Mask out diagonal
    grid_exc[:, np.arange(N), np.arange(N)] = neg_inf
    if keep_mask is not None:
        grid_exc[:, ~keep_mask] = neg_inf

    flat_grid_exc = grid_exc.view(grid.size(0), -1)

    logp_grid_exc = th.nn.functional.log_softmax(flat_grid_exc, dim=-1)
    
    logp_grid_exc = logp_grid_exc.view(grid.size())
    return logp_grid_exc

def mask_grid_asym(grid, pair_set, keep_mask: np.array = None):
    """log-softmax with only valid inputs, excluding zero padded parts"""
    N = grid.size(1)
    grid_exc = grid.clone()
    neg_inf = -np.inf
    for i, j in pair_set:
        grid_exc[:, i, j] = neg_inf

    if keep_mask is not None:
        grid_exc[:, ~keep_mask] = neg_inf

    flat_grid_exc = grid_exc.view(grid.size(0), -1)

    logp_grid_exc = th.nn.functional.log_softmax(flat_grid_exc, dim=-1)
    
    logp_grid_exc = logp_grid_exc.view(grid.size())
    return logp_grid_exc

# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: th.Tensor, mask: th.Tensor, dim: int = -1) -> th.Tensor:
	"""
	``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
	masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
	``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
	``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
	broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
	unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
	do it yourself before passing the mask into this function.
	In the case that the input vector is completely masked, the return value of this function is
	arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
	of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
	that we deal with this case relies on having single-precision floats; mixing half-precision
	floats with fully-masked vectors will likely give you ``nans``.
	If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
	lower), the way we handle masking here could mess you up.  But if you've got logit values that
	extreme, you've got bigger problems than this.
	"""
	if mask is not None:
		mask = mask.float()
		while mask.dim() < vector.dim():
			mask = mask.unsqueeze(1)
		# vector + mask.log() is an easy way to zero out masked elements in logspace, but it
		# results in nans when the whole vector is masked.  We need a very small value instead of a
		# zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
		# just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
		# becomes 0 - this is just the smallest value we can actually use.
		vector = vector + (mask + 1e-45).log()
	return th.nn.functional.log_softmax(vector, dim=dim)


def parse_lr(lr_str):
    # Example "constant(1e-4)"
    lr_type, lr_val = lr_str.split("(")
    lr_val = float(lr_val[:-1])
    return Scheduler(lr_val, lr_type)

def safe_mean(arr, default=0):
    if len(arr) == 0:
        return default
    return np.mean(arr)

def radix_comp(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Binary numbers must have the same length")

    for bit1, bit2 in zip(v1, v2):
        if bit1 > bit2:
            return 1  # num1 is larger
        elif bit1 < bit2:
            return -1  # num2 is larger

    return 0  # Numbers are equal



def get_opt():
    parser = argparse.ArgumentParser()
    # General args
    parser.add_argument("--project_name", type=str, default="NeuRes Supervised")
    parser.add_argument("--exp_name", type=str, default="default_exp")
    parser.add_argument("--exp_desc", type=str, default="")
    parser.add_argument("--variant", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="res-3")
    parser.add_argument("--sat_only", action='store_true')
    parser.add_argument("--test_split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--lr_schedule", type=str, default="linear(1e-4)")
    parser.add_argument("--finetune_schedule", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--no_valid", action='store_true')
    parser.add_argument("--val_freq", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--max_H", type=int, default=None)
    parser.add_argument("--max_pLen", type=int, default=None)
    parser.add_argument("--track_VAs", action='store_true')
    parser.add_argument("--eval_dir", type=str, default="default_eval")
    
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_all_checkpoints", action='store_true')
    # Freeze list
    parser.add_argument("--freeze", type=str, default="")
    # Embedder args
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_rounds", type=str, default="32")
    parser.add_argument("--parent_merge", action='store_true')
    parser.add_argument("--partial_round_alpha", type=float, default=1.0)
    parser.add_argument("--relu_act", action='store_true')
    parser.add_argument("--layer_norm", action='store_true')
    

    # Base policy args
    parser.add_argument("--no_recurrence", action='store_true')
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--enc_dec_cycle", action='store_true')
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--predict_sat", action='store_true')
    parser.add_argument("--predict_only", action='store_true')

    # Clause selector args
    parser.add_argument("--c_selector", type=str, default="full_attn")
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--blend_context", action='store_true')
    parser.add_argument("--attn_QK", type=str, default="Emb")
    parser.add_argument("--sparse_attn", action='store_true')
    parser.add_argument("--L_aggregate", type=str, default="avg_in")
    parser.add_argument("--adapt_sat_gt", action='store_true')
    parser.add_argument("--dual_loss_mode", type=str, default="min")
    parser.add_argument("--no_res", action='store_true')
    parser.add_argument("--res_sat_loss", action='store_true')
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topk_train", action='store_true')
    parser.add_argument("--mask_mode", type=str, default="upper")
    parser.add_argument("--mp_per_res", type=int, default=1)
    parser.add_argument("--step_attn_cond", type=str, default="concat")
    parser.add_argument("--C_aggr", type=str, default="sum")

    # Agent args
    parser.add_argument("--solver", type=str, default="NeuRes")
    parser.add_argument("--critic", action='store_true')
    parser.add_argument("--teacher", type=str, default="BC")
    parser.add_argument("--n_episodes", type=int, default=100000)
    parser.add_argument("--reward", type=str, default="binary")
    parser.add_argument("--penalize_timeout", action='store_true')
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--bootstrap_shrink", action='store_true')
    parser.add_argument("--shrink_skip", action='store_true')

    # Data splits
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)


    # Loss args
    parser.add_argument("--res_loss", action='store_true')

    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--test_only", action='store_true')

    opt, _ = parser.parse_known_args()
    # opt = parser.parse_args()

    opt.lr_schedule = parse_lr(opt.lr_schedule)

    if opt.freeze != "":
        opt.freeze = opt.freeze.split(",")
    else:
        opt.freeze = []

    if opt.wandb_key is None:
        # Read wandb API key from file
        with open("wandb_api_key.txt", "r") as f:
            opt.wandb_key = f.read().strip()

    return opt