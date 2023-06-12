import numpy as np
import os
import ntpath
import time

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.saved = False
        self.log_name = './result/opt.txt'
        

    def reset(self):
        self.saved = False


    def eqt_load_model (self):
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            message = '---- EQT model is loaded (%s) \n' % now
            print(message)
            log_file.write('%s\n' % message)
    
    def eqt_load_check_point (self):
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            message = '---- Checkpoint is loaded (%s) \n' % now
            print(message)
            log_file.write('%s\n' % message)

    def eqt_start (self):
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            message = '---- Extracting picks from EQT started (%s.pkl) \n' % now
            print(message)
            log_file.write('%s\n' % message)

    def eqt_end(self):
        message = '---- EQT Picks file (%s) has been generated in the result folder' % self.opt.suffix_EQT


        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            end = time.strftime("%c")
            log_file.write('================ Extracting picks from EQT ended (%s) ================\n' % end)
