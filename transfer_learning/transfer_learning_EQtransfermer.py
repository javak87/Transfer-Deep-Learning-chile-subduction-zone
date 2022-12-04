import os

import matplotlib.pyplot as plt
import numpy as np
import obspy
import optuna
import pandas as pd
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.dates import epoch2num
from optuna.trial import TrialState
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from data_preprocessing import Data_Preprocessing
from picks_comparison import Picks_Comparison
from transfer_learning import Phasenet_Transfer_Learning

writer = SummaryWriter('runs/EQT')

class EQTransformer_Transfer_Learning(object):

    '''
    This calss perform a transfer learining using "EQ Transformer" network.
    The orginal paper can be find in the follwoing link:

    https://www.nature.com/articles/s41467-020-17591-w
    '''

    def __init__ (self, base_path:'str', base_model:'str', loss_weight:'list', lr:'float', 
                    batch_size:'int', num_workers:'int', 
                    epoch:'int', add_scal=True, add_hist=True):

                    '''
                    Parameters initialization:
                        - base_path : The path of training data.
                                            This directory must contain 'metadata.csv' and 'waveform.hdf5'.
                        
                        - base_model: The name of base model. 
                            The existed base model are:
                                - 'instance'
                                - 'ethz'
                                - 'scedc'
                                - 'geofon'
                                - 'neic'

                        - loss_weight: A list of weights belogs to P picks, S picks and Noise respectively
                        - lr: learning rate
                        - batch_size: batch size
                        - epoch: number of epoch
                        - add_scal: bool variable for visualization metrics and loss function using Tenserboard.
                                If add_scal== True, training loss, f1score, recall, and pres of both P picks and S picks
                                    will be shown in the Phasenet/run folder. The created file can be opened using tensorborad.

                        - add_hist: bool variable for visualization Phasnet weights using Tenserboard.
                                If add_scal== True, phasenet weights during learning.
                                    will be shown in the EQ_Trasfermer/run folder. The created file can be opened using tensorborad.
                    '''

                    self.base_model = base_model
                    self.base_path = base_path
                    self.loss_weight = loss_weight
                    self.lr = lr
                    self.batch_size=batch_size
                    self.num_workers=num_workers
                    self.epoch=epoch

                    # Switch device to GPU if GPU avialable. 
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

                    #assert isinstance(['instance', 'ethz', 'scedc', 'geofon', 'neic'], self.base_model) # The given base model is not existing.

                    print(f" You are using : {self.device} to train the model")
    
    def __call__ (self)-> None:


        # Load the pre-tarined model 
        model = self.load_model()

        
        # Because cross entropy loss will be utilized as loss function, 
        # softmax is used inside of cross entropy loss, softmax should remove in this step.

        #self.remove_softmax(model)

        # divide data into train, validation,and test set.
        train_loader,_, test_loader = self.data_loader(self.base_path, self.batch_size, self.num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        #weights = torch.FloatTensor(self.loss_weight).to(self.device)
        #criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.BCELoss()

        for t in range(self.epoch):
            print(f"Epoch {t+1}\n +++++++++++++++++++++++++++++")
            model.train()
            self.train_loop(model, train_loader, t, optimizer, criterion)
            print('------- training performance -------')
            _,_ = self.check_accuracy(train_loader, model, t)
            model.eval()
            
            print('------- Test performance -------')
            self.test_loop(model, test_loader,t, criterion)
            _,_ = self.check_accuracy(test_loader, model, t)

            # save check_point
            if t % 5 == 0:

                self.save_checkpoint(model)

        writer.close()

    def load_model (self):
        '''
        Load the pre-trained EQTransformer based on the given base_model name.
        '''
        model = sbm.EQTransformer.from_pretrained(self.base_model).to(device=self.device)
    
        return model
    
    @staticmethod
    def remove_softmax(model):
        '''
        This function find the softmax activation function and replaced by Identity.
        By using this function softmax activation function removed from the last layer.
        Parameter:
                    - model: pre-trained model
        '''

        for child_name, child in model.named_children():
            if isinstance(child, nn.Softmax):
                setattr(model, child_name, nn.Identity())
            else:
                Phasenet_Transfer_Learning.remove_softmax(child)

    @staticmethod
    def data_loader (base_path:'str', batch_size:'int', num_workers:'int'):

        '''
        This function used to load waveform into memory.
        Parameter:
                    - base_path: the path of training data.
                                This directory must contain 'metadata.csv' and 'waveform.hdf5'.
                    - batch_size: batch size
                    - num_workers: number of workers
        '''
        

        data = sbd.WaveformDataset(base_path, sampling_rate=100, cache='trace')
        data.preload_waveforms(pbar=True)

        # Divide data into train,dev, and test set.
        train,dev,test = data.train_dev_test()
        #train.preload_waveforms(pbar=True)

        #data_iqu = sbd.WaveformDataset('/home/javak/.seisbench/datasets/iquique', sampling_rate=100, cache='trace')
        #data_iqu.preload_waveforms(pbar=True)

        # Divide data into train,dev, and test set.
        #data_iqu.preload_waveforms(pbar=True)

        #train,dev,_= data_iqu.train_dev_test()
        #base_path_day1='/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/Creat_datasets/day1'
        #data_day1 = sbd.WaveformDataset(base_path_day1, sampling_rate=100, cache='trace')
        #data_day1.preload_waveforms(pbar=True)
        #train= data_day1.train()

        #base_path_day2='/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/Creat_datasets/day2'
        #data_day2 = sbd.WaveformDataset(base_path_day2, sampling_rate=100, cache='trace')
        #data_day2.preload_waveforms(pbar=True)
        #test= data_day2.test()


        phase_dict = {
            "trace_P_arrival_sample": "P",
            "trace_S_arrival_sample": "S",
        }

        # add generator to train, dev, and test dataset
        train_generator = sbg.GenericGenerator(train)
        dev_generator = sbg.GenericGenerator(dev)
        test_generator = sbg.GenericGenerator(test)

        # Define augmentation
        augmentations = [
            sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=9000, selection="random", strategy="variable"),
            sbg.RandomWindow(windowlen=6000, strategy="pad"),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
            #sbg.Filter(N=3, Wn=torch.tensor([1.2, 6], dtype=torch.float32), btype='bandpass'),
            sbg.ChangeDtype(np.float32),
            sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=25, dim=0),
            #sbg.GaussianNoise(scale=(0, 0.2)),
            #sbg.ChannelDropout(),
            #sbg.AddGap()
        ]

        # add defined augmetations to data 
        train_generator.add_augmentations(augmentations)
        dev_generator.add_augmentations(augmentations)
        test_generator.add_augmentations(augmentations)              

        # define data loader

        train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
        dev_loader = DataLoader(dev_generator, batch_size =batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
        test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)

        return train_loader,dev_loader, test_loader 
    
    @staticmethod
    def train_loop(model, dataloader, current_epo:'int', optimizer, criterion,add_scal=True, add_hist=True ):
        

        train_loss = 0
        size = len(dataloader.dataset)
        i = 0



        for batch_id, batch in enumerate(dataloader):

            # Compute prediction        
            pred_eqt = model(batch["X"].to(model.device))
            
            pred = torch.stack([pred_eqt[1], pred_eqt[2], 1-pred_eqt[0]], dim=1)
            # find the the argmax of ground truth
            idx = torch.argmax(batch["y"], dim=1, keepdims=True)

            batch["y"] = torch.zeros_like(batch["y"]).scatter_(1, idx, 1.)

            # compute loss function
            loss_p = criterion(pred[:,0,:].float(), batch["y"][:,0,:].to(model.device).float())
            loss_s = criterion(pred[:,1,:].float(), batch["y"][:,1,:].to(model.device).float())
            loss_n = criterion(pred[:,2,:].float(), batch["y"][:,2,:].to(model.device).float())
            loss = loss_p + loss_s + loss_n
            #loss= criterion(pred.float(), batch["y"].to(model.device).float())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss +=  loss.item()


            if batch_id % 10 == 0:
                loss, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            i += 1
        print(f"Train avg loss: {train_loss/i:>8f} \n")
        
        if add_scal == True:
            writer.add_scalar('Training avg loss', train_loss/i, global_step=current_epo)


    def test_loop(self, model, dataloader, current_epo:'int',  criterion, add_scal=True):
        test_loss = 0
        j=1
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                pred_eqt= model(batch["X"].to(model.device))
                
                pred = torch.stack([pred_eqt[1], pred_eqt[2], 1-pred_eqt[0]], dim=1)

                idx_test = torch.argmax(batch["y"], dim=1, keepdims=True)
                batch["y"] = torch.zeros_like(batch["y"]).scatter_(1, idx_test, 1.)

                loss_p = criterion(pred[:,0,:].float(), batch["y"][:,0,:].to(model.device).float())
                loss_s = criterion(pred[:,1,:].float(), batch["y"][:,1,:].to(model.device).float())
                loss_n = criterion(pred[:,2,:].float(), batch["y"][:,2,:].to(model.device).float())
                test_loss += loss_p + loss_s + loss_n
                j+=1
        print(f"Test avg loss: {test_loss/j:>8f} \n")

        if add_scal == True:
            writer.add_scalar('Test avg loss', test_loss/j, global_step=current_epo)

        return test_loss
    
    @staticmethod
    def check_accuracy(loader, model,current_epo, add_scal = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        correct_true_p = 0
        target_true_p = 0
        predicted_true_p = 0

        correct_true_s = 0
        target_true_s = 0
        predicted_true_s = 0
        
        model.eval()

        with torch.no_grad():
            for batch in loader:
                y = batch['y'].to(device)
                idx = torch.argmax(y, dim=1, keepdims=True)
                Y = torch.zeros_like(y).scatter_(1, idx, 1.)

                #pred = torch.nn.functional.softmax(model(X), dim=1)
                pred_eqt= model(batch['X'].to(device))
                pred = torch.stack([pred_eqt[1], pred_eqt[2], 1-pred_eqt[0]], dim=1)

                idx_ = torch.argmax(pred, dim=1, keepdims=True)
                pred = torch.zeros_like(pred).scatter_(1, idx_, 1.)
                
                correct_true_p += (pred[:,0,:] * Y[:,0,:]).sum()
                target_true_p += (Y[:,0,:]).sum()
                predicted_true_p += (pred[:,0,:]).sum()

                correct_true_s += (pred[:,1,:] * Y[:,1,:]).sum()
                target_true_s += (Y[:,1,:]).sum()
                predicted_true_s += (pred[:,1,:]).sum()
        
        recall_p = correct_true_p/target_true_p
        recall_s = correct_true_s/target_true_s
        precision_p = correct_true_p/predicted_true_p
        precision_s = correct_true_s/predicted_true_s

        f1_score_p = 2 * precision_p * recall_p/ (precision_p + recall_p)
        f1_score_s = 2 * precision_s * recall_s/ (precision_s + recall_s)

        print(
            f" P f1_score: {f1_score_p*100:.3f}\n"
            f" P recall: {recall_p*100:.3f}\n"
            f" P precision: {precision_p*100:.3f}\n"
        )

        print(
            f" S f1_score: {f1_score_s*100:.3f}\n"
            f" S recall: {recall_s*100:.3f}\n"
            f" S precision: {precision_s*100:.3f}\n"
        )

        model.train()

        if add_scal == True:
            writer.add_scalar('P picks Precision', precision_p, global_step=current_epo)
            writer.add_scalar('S picks Precision', precision_s, global_step=current_epo)
            writer.add_scalar('P Recall', recall_p, global_step=current_epo)
            writer.add_scalar('S Recall', recall_s, global_step=current_epo)
            writer.add_scalar('P f1_score', f1_score_p, global_step=current_epo)
            writer.add_scalar('S f1_score', f1_score_s, global_step=current_epo)
        return f1_score_p, f1_score_s
    
    def save_checkpoint(self, model, file_name='transfer_learing_EQT.pth.tar'):
        print('==> saving check_point')
        torch.save(model.state_dict(), file_name)

    
class Annotation(object):
    def __init__(self):
        
        # load model
        model = sbm.EQTransformer.from_pretrained('instance').to(device="cuda" if torch.cuda.is_available() else "cpu")
        #checkpoint = torch.load('transfer_learing_EQT.pth.tar')
        #model.load_state_dict(checkpoint)
        self.model = model

    def deploy_model(self, start_year_analysis, start_day_analysis,
                            end_year_analysis, end_day_analysis,
                            P_th=0.075, S_th=0.1):

        obj = Data_Preprocessing (start_year_analysis, start_day_analysis,
                            end_year_analysis, end_day_analysis)

                
        stream = obj.get_waveforms_chile()

        picks,_ = self.model.classify(stream, batch_size=256, P_threshold=P_th, S_threshold=S_th, parallelism=1)

        pick_df = []
        for p in picks:
            pick_df.append({
                "id": p.trace_id,
                "timestamp": p.peak_time.datetime,
                "prob": p.peak_value,
                "type": p.phase.lower()
            })

        automatic_picks = pd.DataFrame(pick_df)
        PhaseNet_result_p_picks = automatic_picks[automatic_picks.type =='p']
        PhaseNet_result_s_picks = automatic_picks[automatic_picks.type =='s']
        PhaseNet_result_p_picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog', 'PhaseNet_result_p_picks.pkl'))
        PhaseNet_result_s_picks.to_pickle(os.path.join('/home/javak/Sample_data_chile/Comparing PhaseNet and Catalog', 'PhaseNet_result_s_picks.pkl'))
       
        return automatic_picks



class Parameters_Tuning (object):

    '''
    This class is used to tune the phasenet model params with the help of Optuna.
    
    Optuna is a software framework for automating the optimization process of these hyperparameters.
    It automatically finds optimal hyperparameter values by making use of different samplers 
    such as grid search, random, bayesian, and evolutionary algorithms.
    '''
    def __init__ (self, base_path:'str', base_model_list:'list', 
                    loss_weight_range:'list', batch_size_list:'list'):

        '''
        Parameters:
                - base_path: the path of training data.
                        This directory must contain 'metadata.csv' and 'waveform.hdf5'.
                - base_model_list: the list of base_model
                - loss_weight_range: the maximum and minimum range of loss weights
                - batch_size_list: the list of batch size list
        '''
        self.base_path = base_path
        self.base_model_list = base_model_list
        self.loss_weight_range = loss_weight_range
        self.batch_size_list = batch_size_list


    def __call__ (self):

        # create optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, timeout=600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial):

        # Generate the model.
        model_name = trial.suggest_categorical("model_name", self.base_model_list)
        model = sbm.EQTransformer.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        

        self.batch_size = trial.suggest_int("self.batch_size", self.batch_size_list[0], self.batch_size_list[1], step=128)
        _,dev_loader, _ = EQTransformer_Transfer_Learning.data_loader(self.base_path, self.batch_size, num_workers=10)
        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        #weight_p = trial.suggest_float("weight_p", self.loss_weight_range[0], self.loss_weight_range[1])
        #weight_s = trial.suggest_float("weight_s", self.loss_weight_range[0], self.loss_weight_range[1])
        #weight_n = trial.suggest_float("weight_n", self.loss_weight_range[0], self.loss_weight_range[1])
        #criterion = nn.BCELoss(weight=torch.FloatTensor([weight_p, weight_s, weight_n]).to(model.device))
        criterion = nn.BCELoss()

        for t in range(50):
            print(f"Epoch {t+1}\n +++++++++++++++++++++++++++++")
            EQTransformer_Transfer_Learning.train_loop(model, dev_loader, t, optimizer, criterion)

            print('------- training performance -------')
            model.eval()
            _, f1_score_s = EQTransformer_Transfer_Learning.check_accuracy(dev_loader, model, t)
                        

            trial.report(f1_score_s, t)
            


            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return f1_score_s

if __name__ == '__main__':


    start_year_analysis = 2007
    start_day_analysis = 304
    end_year_analysis = 2007
    end_day_analysis = 304

    obj = Annotation()
    obj.deploy_model(start_year_analysis, start_day_analysis, end_year_analysis, end_day_analysis)
    
    '''
    base_path ='/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/Creat_datasets/day1'
    base_model = 'instance'
    loss_weight = [2.22, 4.28, 3.4]
    lr = 0.0006
    batch_size = 256
    num_workers = 10
    epoch =70
    
    phasenet_obj = EQTransformer_Transfer_Learning(base_path, base_model,
                            loss_weight,
                            lr,
                            batch_size, num_workers, 
                            epoch, 
                            add_scal=True, add_hist=True)


    phasenet_obj()

    
    base_path ='/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/Creat_datasets/day1'

    base_model_list = ['instance', 'original']
    loss_weight_range = [2,10]
    batch_size_list = [128, 512]
    tun_obj = Parameters_Tuning (base_path, base_model_list, loss_weight_range, batch_size_list)
    tun_obj()
    '''
    

    