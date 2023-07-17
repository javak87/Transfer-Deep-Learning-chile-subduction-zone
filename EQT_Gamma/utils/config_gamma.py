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


        ## Eikonal for 1D velocity model
        #zz = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 130, 150, 180, 210, 250]
        #vp = [5.87, 5.43, 6.84, 6.93, 7.00, 6.89, 8.49, 7.65, 7.80, 8.44, 7.62, 8.75, 8.49, 8.49, 8.49, 8.49]
        zz = [2.5, 10.50, 40, 70, 100, 200]
        vp = [5.37, 5.98, 7.41, 8.48, 8.49, 8.50]
        vp_vs_ratio = 1.73
        vs = [v / vp_vs_ratio for v in vp]
        #h = 0.3
        h = 1
        vel = {"z": zz, "p": vp, "s": vs}
        config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

        # Filtering
        config["min_picks_per_eq"] = self.opt.min_picks_per_eq
        config["min_p_picks_per_eq"] = self.opt.min_p_picks_per_eq
        config["min_s_picks_per_eq"] = self.opt.min_s_picks_per_eq

        config["max_sigma11"] = self.opt.max_sigma11
        config["max_sigma22"] = self.opt.max_sigma22
        config["max_sigma12"] = self.opt.max_sigma12 
        
        config["covariance_prior"] = self.opt.covariance_prior 

        return config      

    
