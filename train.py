from envs import ResUNSAT
import torch as th
# import cupy as cp
import numpy as np
from trainers.teacher_force import TeacherForce
from common.utils import get_config
import random

config = get_config()
th.manual_seed(config['seed'])
# cp.random.seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])

env = ResUNSAT(config, shuffle=True)

trainer = TeacherForce(config, env)

trainer.train(n_epochs_plan=config['n_epochs'],  n_epochs_stop=config['n_epochs'])

