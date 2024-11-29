from envs import ResUNSAT
import torch as th
# import cupy as cp
import numpy as np
from trainers.teacher_force import TeacherForce
from common.utils import get_opt
import random

opt = get_opt()
th.manual_seed(opt.seed)
# cp.random.seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

env = ResUNSAT(opt, shuffle=True)

trainer = TeacherForce(opt, env)

trainer.train(n_epochs_plan=opt.n_epochs,  n_epochs_stop=opt.n_epochs, tag="supervised")

