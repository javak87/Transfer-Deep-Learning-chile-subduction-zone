import sys
import os
import time
import torch
import seisbench.models as sbm
from pathlib import Path
from options.EQT_options import EQTOptions
from utils.df_file_path_generator import generate_DF_file_path
from utils.data_preprocessing import Data_Preprocessing
import pandas as pd



if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = EQTOptions().parse()

    opt.suffix_EQT = (opt.suffix_EQT + 
                      '_bas_modl:' + opt.base_model
                    + '_chk_pt:' + opt.check_point
                    +'_p_th:' + str(opt.P_threshold)
                    +'_s_th:' + str(opt.S_threshold)
                    +'_det_th:' + str(opt.detection_threshold))
    # Load model
    model = sbm.EQTransformer.from_pretrained(opt.base_model).to(device="cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('./models/'+ opt.check_point)
    model.load_state_dict(checkpoint)

    # Generate DF_chile_path_file.pkl files if is not existing
    if opt.file_path_generator == True:
        generate_DF_file_path(opt.data_path, os.path.join(Path(os.path.dirname(__file__)), 'result/df_path_files'))

    # Create a stream object
    stream = Data_Preprocessing(opt.time_interval[0],opt.time_interval[1],opt.time_interval[2],opt.time_interval[3]).get_waveforms_chile()

    # Extract picks from EQT
    picks,_ = model.classify(stream, batch_size=opt.batch_size, P_threshold=opt.P_threshold, S_threshold=opt.S_threshold, detection_threshold=opt.P_threshold, parallelism=1)
    
    pick_df = []
    for p in picks:
        pick_df.append({
            "id": p.trace_id,
            "timestamp": p.peak_time.datetime,
            "prob": p.peak_value,
            "type": p.phase.lower()
        })

    automatic_picks = pd.DataFrame(pick_df)
    automatic_picks.to_pickle(os.path.dirname(__file__) + '/result/picker_output/' + opt.suffix_EQT + '.pkl')
    #picker_result_p_picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog', 'PhaseNet_result_p_picks.pkl'))
    #picker_result_s_picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog', 'PhaseNet_result_s_picks.pkl'))
    '''
    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                NiftiDataset.Resample(opt.new_resolution, opt.resample),
                NiftiDataset.Augmentation(),
                NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
                NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
                ]

    train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=True, train=True)
    print('lenght train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        '''









