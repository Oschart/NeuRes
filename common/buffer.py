from common.logger import Logger


class RolloutBuffer:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(config)

    def reset(self):
        for mode in self.logger.stats_buffer:
            self.logger.stats_buffer[mode] = {"Iterations": 0, "Samples": 0}

    def add_rollout_stat(self, log_dict, mode, tag, ep_count, roll_window=20):
        # Logger only knows "train" and "test"; route validation through "test".
        logger_mode = "train" if mode == "train" else "test"
        self.logger.record(log_dict, logger_mode, tag, ep_count, roll_window=roll_window)
