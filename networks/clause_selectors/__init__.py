from networks.clause_selectors.full_attn import FullAttention
from networks.clause_selectors.casc_attn import CascAttention
from networks.clause_selectors.anch_attn import AnchAttention


attn_module_map = {
    "full_attn": FullAttention,
    "casc_attn": CascAttention,
    "anch_attn": AnchAttention,
}

def make_attn_module(attn_module_name, config):
    return attn_module_map[attn_module_name](config)

