import torch as th
import numpy as np
from common.logger import Logger
from collections import deque


class RolloutBuffer:
    def __init__(self, opt):
        self.opt = opt
        self.logger = Logger(opt)
        self.modes_every = {"train": self.opt.log_every, "valid_train": 1, "valid_val": 1}
        self.reset()

    def reset(self):
        self.stats_buffer = {}
        for mode in self.modes_every:
            self.stats_buffer[mode] = {"step": 0}
    
    def add_rollout_stat(self, log_dict, mode, tag, ep_count, roll_window=20):
        for key, val in log_dict.items():
            if isinstance(val, dict):
                for subkey, v in val.items():
                    subkey = f"{key}/{subkey}"
                    if subkey not in self.stats_buffer[mode]:
                        self.stats_buffer[mode][subkey] = deque(maxlen=roll_window)
                    self.stats_buffer[mode][subkey].append(v)
            else:
                if key not in self.stats_buffer[mode]:
                    self.stats_buffer[mode][key] = deque(maxlen=roll_window)
                self.stats_buffer[mode][key].append(val)
        step = self.stats_buffer[mode]["step"]

        # if step % self.modes_every[mode] == 0:
        prefix = f"{tag}/{mode}"
        log_dict_ = {f"{prefix}/{k}": np.mean(v) for k,v in self.stats_buffer[mode].items()}
        log_dict_["episodes"] = ep_count
        self.logger.log(log_dict_, step)

        self.stats_buffer[mode]["step"] += 1
