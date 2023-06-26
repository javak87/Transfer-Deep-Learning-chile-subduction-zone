import argparse
import os
from pathlib import Path
#from utils.utils import *
#import torch
#import models


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--data_path', type=str, default='/data2/chile/CHILE_COMBINED_2021', help='path of stored sensor data')
        parser.add_argument('--file_path_generator', type=bool, default=False, 
                            help='if you generate DF_chile_path_file.pkl for all files you don"t need to generate another one')

        parser.add_argument('--time_interval', type=tuple, default=(2011,1,2011,1), 
                            help='The time interval for styd.(a1: start_year,a2: start_day,a3: end_year,a4: end_day)')
        #parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

        parser.add_argument('--client', type=str, default='GFZ', help='FDSN client name')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        parent_dir = Path(os.path.dirname(__file__)).parent
        expr_dir = os.path.join(parent_dir, 'result')
        if os.path.exists(expr_dir) == False:
            os.mkdir(expr_dir)
        
        current_dir = Path(os.path.dirname(__file__)).parent
        path_files_dir = os.path.join(current_dir, 'result/df_path_files')
        if os.path.exists(path_files_dir) == False:
            os.mkdir(path_files_dir)
        
        picker_out_dir = os.path.join(current_dir, 'result/picker_output')
        if os.path.exists(picker_out_dir) == False:
            os.mkdir(picker_out_dir)
        
        snuffler_out_dir = os.path.join(current_dir, 'result/snuffler_output')
        if os.path.exists(snuffler_out_dir) == False:
            os.mkdir(snuffler_out_dir)
        
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        #opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        #if opt.suffix:
        #    suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #    opt.name = opt.name + suffix

        self.print_options(opt)

        self.opt = opt
        return self.opt

