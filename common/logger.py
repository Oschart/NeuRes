import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger():
    def __init__(self, opt, use_tb=False, use_wandb=True):
        self.opt = opt
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        if use_wandb:
            self.setup_wandb(opt)
        if use_tb:
            self.tb_writer = SummaryWriter(f"logs/{opt.exp_name}")
        
        self.log_keys = {"episodes"}

    def setup_wandb(self, opt):
        wandb.login(key=opt.wandb_key)
        run = wandb.init(
            project=f"{opt.project_name}",
            group=f"{opt.exp_name}_{opt.variant}",
            notes=opt.exp_desc,
            name=opt.variant, 
            config=opt
        )
        wandb.define_metric("episodes")
        # define a metric we are interested in the minimum of
        wandb.define_metric("train_supervised/loss", summary="min")
        # define a metric we are interested in the maximum of
        wandb.define_metric("valid_val_supervised/solved", summary="max")
        wandb.define_metric("valid_val_supervised/optimality", summary="max")
    
    def log_tb(self, log_dict, step):
        for tag, value in log_dict.items():
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_wandb(self, log_dict, step):
        # wandb.log(log_dict, step=step)
        for key in log_dict:
            if key not in self.log_keys:
                self.log_keys.add(key)
                wandb.define_metric(key, step_metric="episodes")
        wandb.log(log_dict)

    def log(self, log_dict, step):
        if self.use_tb:
            self.log_tb(log_dict, step)
        if self.use_wandb:
            self.log_wandb(log_dict, step)
