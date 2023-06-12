import numpy as np
import os
import ntpath
import time

class ConfigGamma():
    def __init__(self, opt):
        self.opt = opt

    def __call__(self):
        config = self.create_config()
        return config

    def create_config (self):

        # Gamma
        config = {}
        config["dims"] = ['x(km)', 'y(km)', 'z(km)']
        config["use_dbscan"] = self.opt.use_dbscan
        config["use_amplitude"] = self.opt.use_amplitude
        config["x(km)"] = self.opt.x_interval
        config["y(km)"] = self.opt.y_interval
        config["z(km)"] = self.opt.z_interval
        config["vel"] = {"p": 7.0, "s": 7.0 / 1.75}  # We assume rather high velocities as we expect deeper events
        config["method"] = self.opt.method
        config["oversample_factor"] = self.opt.oversampling_factor

        # DBSCAN
        config["bfgs_bounds"] = (
            (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
            (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
            (0, config["z(km)"][1] + 1),  # x
            (None, None),  # t
        )
        config["dbscan_eps"] = self.opt.dbscan_eps  # seconds
        config["dbscan_min_samples"] = self.opt.dbscan_min_samples

        # Filtering
        config["min_picks_per_eq"] = self.opt.min_picks_per_eq
        config["max_sigma11"] = self.opt.max_sigma11
        config["max_sigma22"] = self.opt.max_sigma22
        config["max_sigma12"] = self.opt.max_sigma12 

        return config      

    
