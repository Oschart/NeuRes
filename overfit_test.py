# Overfit on one episode
from envs import ResUNSAT
import torch as th
import numpy as np
from trainers.teacher_force import TeacherForce
import pickle as pkl
import argparse
import wandb
from common.logger import Logger

from common.utils import get_opt

opt = get_opt()
th.manual_seed(opt.seed)
np.random.seed(opt.seed)

env = ResUNSAT(opt, shuffle=False, overfit=True)
trainer = TeacherForce(opt, env)

overfit_steps = 1000
def callback(epoch, avg_solved):
    if avg_solved == 1.0:
        print(f"Overfit test PASSED! Solved in {epoch+1} epochs.")
        exit(0)
    

# Supervised pretraining
trainer.train(n_epochs_plan=overfit_steps, n_epochs_stop=overfit_steps, tag="supervised", callback=callback)
print(f"Overfit test FAILED!")