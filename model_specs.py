from torchsummary import summary
from envs import ResUNSAT
from solvers import BaseSolver
from common.utils import get_opt
import json
from tabulate import tabulate



def count_parameters(model):
	table = {"Modules": [], "Shape": [], "Parameters": []}
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: continue
		params = parameter.numel()
		table["Modules"] += [name]
		table["Parameters"] += [f"{params:,}"]
		shape = "x".join(str(x) for x in list(parameter.shape))
		table["Shape"] += [shape]
		total_params+=params
	# print(table)
	print(f"Total Trainable Params: {total_params}")
	table["Modules"] += ["Total"]
	table["Parameters"] += [f"{total_params:,}"]
	latex_table = tabulate(table, headers='keys', tablefmt='latex_raw')
	return latex_table

def update_opts(opts, new_opts):
	for k, v in new_opts.items():
		setattr(opts, k, v)
	return opts

# Load model registry JSON
with open("ablations.json", "r") as f:
	model_registry = json.load(f)

model_name = "CA-Dynamic"
model_opts = model_registry[model_name]
opt = update_opts(get_opt(), model_opts)

env = ResUNSAT(opt, shuffle=False)
formula = env.reset()
solver = BaseSolver(opt, env)
solver.reset()
# solver.setup_formula(formula)
s = count_parameters(solver)

replace_map = {
	"policy.": "",
	"rnn.": "",
	"L_init": "L_{init}",
	"C_init": "C_{init}",
	"L_update": "L_{update}",
	"C_update": "C_{update}",
	"L_msg": "L_{msg}",
	"C_msg": "C_{msg}",
	"c_selector": "C_{selector}",
	"_attn": "_{attn}",
	
	"_ih_l0_reverse": "_{ih-l0-rev}",
	"_hh_l0_reverse": "_{hh-l0-rev}",
	"_ih_l0": "_{ih-l0}",
	"_hh_l0": "_{hh-l0}",
	"_ih": "_{ih}",
	"_hh": "_{hh}",
	"Total": "\hline \\textbf{Total}",
	"Modules": "\\textbf{Modules}",
	"Shape": "\\textbf{Shape}",
	"Parameters": "\\textbf{\#Parameters}",
}
for k, v in replace_map.items():
	s = s.replace(k, v)

# convert to latex
# s = s.replace(' ', '&')
# s = s.replace('\n', '\\\\')
# s = s.replace('(', '{')
# s = s.replace(')', '}')
# s = s.replace('---', '-')
# s = s.replace('===', '=')

print(s)
