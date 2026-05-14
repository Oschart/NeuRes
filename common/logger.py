import torch as th
import numpy as np
import wandb
from common.utils import safer_mean
from collections import deque
from time import time

class Logger():
    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.use_wandb = not config["no_wandb"]
        self.start_time = time()
        if not config["no_wandb"]:
            self.setup_wandb(config)

        self.log_every = {
            "train": config["log_every"],
            "test": 1
        }
        
        self.log_keys = {"Iterations"}
        self.stats_buffer = {"train": {}, "test": {}}
        self.stats_buffer["train"]["Iterations"] = 0
        self.stats_buffer["test"]["Iterations"] = 0
        self.stats_buffer["train"]["Samples"] = 0
        self.stats_buffer["test"]["Samples"] = 0

    def setup_wandb(self, config):
        wandb.login(key=config["wandb_key"])
        run = wandb.init(
            project=f"{config['project_name']}",
            group=f"{config['exp_name']}_{config['variant']}",
            notes=config["exp_desc"],
            name=config["variant"], 
            config=config
        )
        wandb.define_metric("Iterations")
        # define a metric we are interested in the minimum of


    
    def log_wandb(self, log_dict):
        for key in log_dict:
            if key not in self.log_keys:
                self.log_keys.add(key)
                wandb.define_metric(key, step_metric="Iterations")
        wandb.log(log_dict)

    def log(self, log_dict):
        if self.use_wandb:
            self.log_wandb(log_dict)
    
    def process_dict(self, prefix: str, log_dict: dict, mode: str, roll_window: int = 20):
        # Process nested dicts if any
        for key, val in log_dict.items():
            if isinstance(val, dict):
                prefix_ = f"{prefix}/{key}"
                self.process_dict(prefix_, val, mode, roll_window)
            else:
                if key not in self.stats_buffer[mode]:
                    self.stats_buffer[mode][key] = deque(maxlen=roll_window)
                self.stats_buffer[mode][key].append(val)

    def record(self, log_dict, mode, tag, iteration, roll_window=20):
        self.process_dict("", log_dict, mode, roll_window)

        if iteration % self.log_every[mode] != 0:
            return
        
        prefix = f"{tag}/{mode}" if tag != "" else mode
        log_dict_ = {f"{prefix}/{k}": safer_mean(v) for k,v in self.stats_buffer[mode].items()}
        log_dict_["Iterations"] = iteration
        log_dict_["Samples"] = iteration * self.batch_size
        log_dict_[f"{prefix}/Time (s)"] = time() - self.start_time
        self.log(log_dict_)

