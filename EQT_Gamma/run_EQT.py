import sys
import os
import time
import torch
import seisbench.models as sbm
from pathlib import Path
from options.EQT_options import EQTOptions
from utils.df_file_path_generator import generate_DF_file_path
from utils.data_preprocessing import Data_Preprocessing
from utils.visualizer import Visualizer
import pandas as pd
import pickle



if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = EQTOptions().parse()

    opt.suffix_EQT = (opt.suffix_EQT + 
                      '_bas_modl:' + opt.base_model
                    + '_chk_pt:' + str(opt.activate_check_point)
                    + '_which_chk_pt:' + opt.which_check_point
                    +'_p_th:' + str(opt.P_threshold)
                    +'_s_th:' + str(opt.S_threshold)
                    +'_det_th:' + str(opt.detection_threshold)
                    + '_batch_size:' + str(opt.batch_size))
    
    # start logging
    vis = Visualizer(opt)

    # Load model
    vis.eqt_load_model()
    model = sbm.EQTransformer.from_pretrained(opt.base_model).to(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load check point
    if opt.activate_check_point == True:
        vis.eqt_load_check_point()
        checkpoint = torch.load('./checkpoint/'+ opt.which_check_point)
        model.load_state_dict(checkpoint)

    # Generate DF_chile_path_file.pkl files if is not existing
    if opt.file_path_generator == True:
        generate_DF_file_path(opt.data_path, os.path.join(Path(os.path.dirname(__file__)), 'result/df_path_files'))

    # Create a stream object
    stream = Data_Preprocessing(opt.time_interval[0],opt.time_interval[1],opt.time_interval[2],opt.time_interval[3]).get_waveforms_chile()

    # Extract picks from EQT
    vis.eqt_start()
    picks,_ = model.classify(stream, batch_size=opt.batch_size, P_threshold=opt.P_threshold, S_threshold=opt.S_threshold, detection_threshold=opt.detection_threshold, parallelism=1)
    

    '''
    pick_df = []
    for p in picks:
        pick_df.append({
            "id": p.trace_id,
            "timestamp": p.peak_time.datetime,
            "prob": p.peak_value,
            "type": p.phase.lower()
        })

    #automatic_picks = pd.DataFrame(pick_df)
    automatic_picks.to_pickle(os.path.dirname(__file__) + '/result/picker_output/' + opt.suffix_EQT + '.pkl')
    '''
    with open(os.path.dirname(__file__) + '/result/picker_output/' + opt.suffix_EQT + '.pkl', "wb") as fp:
        pickle.dump(picks, fp)
    vis.eqt_end()








