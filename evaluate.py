from envs import ResUNSAT
import torch as th
import numpy as np
from trainers.teacher_force import TeacherForce
from solvers.base import BaseSolver
from common.utils import PolicySpec, Scheduler
import pickle as pkl
import argparse
from common.utils import get_opt
import random
from torch import multiprocessing as mp
import os
import json

# gpu_count = th.cuda.device_count()

th.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', default='all', help='Which models to test?')
parser.add_argument("--dataset", type=str, default="res-7")
parser.add_argument("--eval_dir", type=str, default="icml_ultimate")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--n_chunks", type=int, default=1)
parser.add_argument("--chunk_id", type=int, default=0)
parser.add_argument("--max_H", type=int, default=None)
parser.add_argument("--track_VAs", action='store_true')
test_opt, _ = parser.parse_known_args()

# Load model registry JSON
with open("ablations.json", "r") as f:
	model_registry = json.load(f)

if test_opt.models == "all":
	test_opt.models = sorted(model_registry.keys())


def update_opts(opts, new_opts):
	for k, v in new_opts.items():
		setattr(opts, k, v)
	return opts


class ModelTest:
	def __init__(self, rank, world_size):
		self.rank = rank
		os.environ['MASTER_ADDR'] = 'localhost'
		os.environ['MASTER_PORT'] = '12355'

		print(f"Models list: {test_opt.models}")

		self.model_name = test_opt.models[rank]
		print(f"Evaluating: {self.model_name}")
		model_opts = model_registry[self.model_name]
		opt = update_opts(get_opt(), model_opts)
		opt.dataset = test_opt.dataset
		opt.eval_dir = test_opt.eval_dir
		opt.test_only = True
		# th.cuda.set_device(opt.device)
		print(f"Testing model: {self.model_name}")
		print("Creating env...")
		print(f"Opts: {opt}")
		env = ResUNSAT(opt, shuffle=False)
		pool_size = len(env.dataset["test"]["problems"])
		print(f"Total test pool = {pool_size}")
		
		print("Creating runner...")
		runner = TeacherForce(opt, env)
		runner.eval_model_name = self.model_name
		addendum = f"_part{test_opt.chunk_id}" if test_opt.n_chunks > 1 else ""
		res = runner.test(addendum)
		os.makedirs(f"{opt.eval_dir}/{opt.dataset}/", exist_ok=True)
		pkl.dump(res, open(f"{opt.eval_dir}/{opt.dataset}/{self.model_name}{addendum}.pkl", "wb"))
		self.print_sat_stats(res)
		# self.print_results(res, buckets)

	def print_sat_stats(self, res):
		solved = np.mean([r.solved for r in res])
		ep_len = np.mean([r.ep_len for r in res if r.solved])
		if test_opt.track_VAs:
			unique_VAs_ratio = np.mean([r.extras["unique_ratio"] for r in res])
		else:
			unique_VAs_ratio = 0
		print("="*30)
		print(f"Model: {self.model_name}")
		print(f"Solved: {solved:.2f}, Ep_len: {ep_len:.2f}, Unique_VAs: {unique_VAs_ratio:.2f}")
		print("="*30)

	
	def print_results(self, res, buckets):
		def get_bucket_id(num_vars):
			for i, (a, b) in enumerate(buckets):
				if num_vars >= a and num_vars <= b:
					return i
			return None
		buck_res = {b: [] for b in buckets}
		for nvars in res:
			buck_id = get_bucket_id(nvars)
			buck_res[buck_id].extend(res[nvars])
		
		print(f"Results for {self.model_name}")
		for b in buck_res:
			solved, extra_steps = zip(*buck_res[b])
			solved_extra_steps = [(s, e) for s, e in buck_res[b] if s]
			solved_avg = np.mean(solved)
			extra_steps_avg = np.mean(solved_extra_steps)
			print(f"Bucket {b}: Success rate: {solved_avg}, p-Len: {extra_steps_avg}")


			
		


if __name__ == "__main__":

	# Check if dataset_test pickle exists
	if not os.path.exists(f"data/{test_opt.dataset}/test_dataset.pkl"): 
		if os.path.exists(f"data/{test_opt.dataset}/dataset.pkl"):
			dataset_dict = pkl.load(open(f"data/{test_opt.dataset}/dataset.pkl", "rb"))
			dataset_test = {"test": dataset_dict['test']}
			pkl.dump(dataset_test, open(f"data/{test_opt.dataset}/test_dataset.pkl", "wb"))

	world_size = len(test_opt.models)
	
	if world_size == 1:
		ModelTest(0, 1)
	else:
		mp.spawn(
			ModelTest,
			nprocs=world_size,
			args=(world_size,),
			join=True)