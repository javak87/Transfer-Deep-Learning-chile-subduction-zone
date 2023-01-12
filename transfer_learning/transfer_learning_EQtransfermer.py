import os
import argparse
import json
import logging
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
import torch.nn.functional as F
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
    # Phase dict for labelling. We only study P and S phases without differentiating between them.
    phase_dict = {
        "trace_P_arrival_sample": "P",
        "trace_S_arrival_sample": "S",
    }

    def __init__ (self, config:'json'):

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
                    self.config = config

                    # Switch device to GPU if GPU avialable. 
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

                    #assert isinstance(['instance', 'ethz', 'scedc', 'geofon', 'neic'], self.base_model) # The given base model is not existing.

                    print(f" You are using : {self.device} to train the model")
    
    def __call__ (self)-> None:


        # Load the pre-tarined model 
        model = self.load_model()

        # load data into memory
        data = self.loading_data_into_mem()
        
        # split data into train, dev, and test
        train,dev,test= EQTransformer_Transfer_Learning.data_spliter(data)

        # create train generator
        train_generator= EQTransformer_Transfer_Learning.data_generator(train)

        # create dev generator
        #dev_generator= EQTransformer_Transfer_Learning.data_generator(dev)

        # create test generator
        test_generator= EQTransformer_Transfer_Learning.data_generator(test)

        # add augmentation to train data
        train_augmentations = self.get_train_augmentations()
        #train_augmentations = self.test_augmentations()

        # add augmentation to dev data
        #dev_augmentations = self.get_train_augmentations()

        # add augmentation to test data
        test_augmentations = self.get_test_augmentations()

        # add defined augmetations to data 
        train_generator.add_augmentations(train_augmentations)
        #dev_generator.add_augmentations(dev_augmentations)
        test_generator.add_augmentations(test_augmentations)


        train_loader = DataLoader(train_generator, batch_size=self.config["trainer_args"]["batch_size"], shuffle=True, num_workers=self.config["trainer_args"]["num_workers"], worker_init_fn=worker_seeding)
        #dev_loader = DataLoader(dev_generator, batch_size =self.config["trainer_args"]["batch_size"], shuffle=False, num_workers=self.config["trainer_args"]["num_workers"], worker_init_fn=worker_seeding)
        test_loader = DataLoader(test_generator, batch_size=self.config["trainer_args"]["batch_size"], shuffle=False, num_workers=self.config["trainer_args"]["num_workers"], worker_init_fn=worker_seeding)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["model_args"]["lr"])
        #criterion = SoftCrossEntropyLossMean()        
        #criterion = SoftCrossEntropyLoss()  
        #criterion= nn.CrossEntropyLoss()
        criterion=nn.BCELoss()

        for t in range(self.config["trainer_args"]["epochs"]):
            print(f"Epoch {t+1}\n +++++++++++++++++++++++++++++")
            model.train()
            EQTransformer_Transfer_Learning.train_loop(config, model, train_loader, t, optimizer, criterion)
            print('------- training performance -------')
            _,_,_ = self.check_accuracy(train_loader, model, t)
            model.eval()
            
            print('------- Test performance -------')
            self.test_loop(model, test_loader,t, criterion)
            _,_,_ = self.check_accuracy(test_loader, model, t)

            # save check_point
            if t % 5 == 0:

                self.save_checkpoint(model)

        writer.close()

    def load_model (self):
        '''
        Load the pre-trained EQTransformer based on the given base_model name.
        '''
        base_model = self.config["model_args"]["base_model"]
        model = sbm.EQTransformer.from_pretrained(base_model).to(device=self.device)
    
        return model

    def loading_data_into_mem (self):

        base_path =self.config["base_path"]
        data = sbd.WaveformDataset(base_path, sampling_rate=100, cache='trace')
        data.preload_waveforms(pbar=True)
        return data 
    

    @staticmethod
    def data_spliter (data):
       
        train,dev,test = data.train_dev_test()
        
        return train,dev,test

    @staticmethod
    def data_generator(data_split):
        return sbg.GenericGenerator(data_split)


    def get_base_augmentations(self):

        p_phases = [key for key, val in EQTransformer_Transfer_Learning.phase_dict.items() if val == "P"]
        s_phases = [key for key, val in EQTransformer_Transfer_Learning.phase_dict.items() if val == "S"]

        if self.config["augmentation"]["detection_fixed_window"] is not None:
            detection_labeller = sbg.DetectionLabeller(
                p_phases,
                fixed_window=self.config["augmentation"]["detection_fixed_window"],
                key=("X", "detections"),
            )
        else:
            detection_labeller = sbg.DetectionLabeller(
                p_phases, s_phases=s_phases, key=("X", "detections")
            )

        block1 = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(EQTransformer_Transfer_Learning.phase_dict.keys()),
                        samples_before=6000,
                        windowlen=12000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=self.config["augmentation"]["sample_boundaries"][0],
                high=self.config["augmentation"]["sample_boundaries"][1],
                windowlen=6000,
                strategy="pad",
            ),
            sbg.ProbabilisticLabeller(
                label_columns=EQTransformer_Transfer_Learning.phase_dict, sigma=self.config["augmentation"]["sigma"], dim=0
            ),
            detection_labeller,
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block2 = [
            sbg.ChangeDtype(np.float32, "X"),
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "detections"),
        ]

        return block1, block2

    def get_train_augmentations(self):

        if self.config["augmentation"]["rotate_array"]:
            rotation_block = [
                sbg.OneOf(
                    [
                        sbg.RandomArrayRotation(["X", "y", "detections"]),
                        sbg.NullAugmentation(),
                    ],
                    [0.99, 0.01],
                )
            ]
        else:
            rotation_block = []

        augmentation_block = [
            # Gaussian noise
            sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [0.5, 0.5]),
            # Array rotation
            *rotation_block,
            # Gaps
            sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [0.2, 0.8]),
            # Channel dropout
            sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [0.3, 0.7]),
            # Augmentations make second normalize necessary
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block1, block2 = self.get_base_augmentations()

        return block1 + augmentation_block + block2

    def get_test_augmentations(self):

        block1, block2 = self.get_base_augmentations()

        return block1 + block2

    def test_augmentations(self):

        p_phases = [key for key, val in EQTransformer_Transfer_Learning.phase_dict.items() if val == "P"]
        s_phases = [key for key, val in EQTransformer_Transfer_Learning.phase_dict.items() if val == "S"]

        if self.config["augmentation"]["detection_fixed_window"] is not None:
            detection_labeller = sbg.DetectionLabeller(
                p_phases,
                fixed_window=self.config["augmentation"]["detection_fixed_window"],
                key=("X", "detections"),
            )
        else:
            detection_labeller = sbg.DetectionLabeller(
                p_phases, s_phases=s_phases, key=("X", "detections")
            )   

        augmentations = [
            sbg.WindowAroundSample(list(EQTransformer_Transfer_Learning.phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
            sbg.RandomWindow(windowlen=6000, strategy="pad"),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
            sbg.ChangeDtype(np.float32),
            sbg.ProbabilisticLabeller(label_columns=EQTransformer_Transfer_Learning.phase_dict, sigma=25, dim=0),
            detection_labeller,
        ]
        return augmentations

    def get_train_augmentations(self):

        p_phases = [key for key, val in EQTransformer_Transfer_Learning.phase_dict.items() if val == "P"]
        s_phases = [key for key, val in EQTransformer_Transfer_Learning.phase_dict.items() if val == "S"]

        if self.config["augmentation"]["detection_fixed_window"] is not None:
            detection_labeller = sbg.DetectionLabeller(
                p_phases,
                fixed_window=self.config["augmentation"]["detection_fixed_window"],
                key=("X", "detections"),
            )
        else:
            detection_labeller = sbg.DetectionLabeller(
                p_phases, s_phases=s_phases, key=("X", "detections")
            )


        block1 = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(EQTransformer_Transfer_Learning.phase_dict.keys()),
                        samples_before=6000,
                        windowlen=12000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low= self.config["augmentation"]["sample_boundaries"][0],
                high=self.config["augmentation"]["sample_boundaries"][1],
                windowlen=6000,
                strategy="pad",
            ),
            sbg.ProbabilisticLabeller(
                label_columns=EQTransformer_Transfer_Learning.phase_dict, sigma=self.config["augmentation"]["sigma"], dim=0
            ),
            detection_labeller,
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block2 = [
            sbg.ChangeDtype(np.float32, "X"),
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "detections"),
        ]


        if self.config["augmentation"]["rotate_array"]:
            rotation_block = [
                sbg.OneOf(
                    [
                        sbg.RandomArrayRotation(["X", "y", "detections"]),
                        sbg.NullAugmentation(),
                    ],
                    [0.99, 0.01],
                )
            ]
        else:
            rotation_block = []

        augmentation_block = [
            # Gaussian noise
            sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [0.5, 0.5]),
            # Array rotation
            *rotation_block,
            # Gaps
            sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [0.2, 0.8]),
            # Channel dropout
            sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [0.3, 0.7]),
            # Augmentations make second normalize necessary
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        train_augmentations = block1 + augmentation_block + block2

        return train_augmentations


    @staticmethod
    def train_loop(config, model, dataloader, current_epo:'int', optimizer, criterion,add_scal=True, add_hist=True ):
        

        train_loss = 0
        size = len(dataloader.dataset)
        i = 0


        for batch_id, batch in enumerate(dataloader):

            # Compute prediction        
            det_pred, p_pred, s_pred = model(batch["X"].to(model.device))
            
            p_true = batch["y"][:, 0].to(model.device)
            s_true = batch["y"][:, 1].to(model.device)
            det_true = batch["detections"][:, 0].to(model.device)

            # compute loss function
            wt = torch.FloatTensor(config["trainer_args"]["loss_weight"]).to(model.device)
            loss = wt[0]*criterion(det_pred.float(), det_true.float())
            + wt[1]*criterion(p_pred.float(), p_true.float())
            + wt[2]*criterion(s_pred.float(), s_true.float())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss +=  loss.item()


            if batch_id % 5 == 0:
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

                # Compute prediction        
                det_pred, p_pred, s_pred = model(batch["X"].to(model.device))
                
                p_true = batch["y"][:, 0].to(model.device)
                s_true = batch["y"][:, 1].to(model.device)
                det_true = batch["detections"][:, 0].to(model.device)

                # compute loss function
                wt = torch.FloatTensor(self.config["trainer_args"]["loss_weight"]).to(model.device)
                test_loss += wt[0]*criterion(det_pred.float(), det_true.float())
                + wt[1]*criterion(p_pred.float(), p_true.float())
                + wt[2]*criterion(s_pred.float(), s_true.float())

                j+=1
        print(f"Test avg loss: {test_loss/j:>8f} \n")

        if add_scal == True:
            writer.add_scalar('Test avg loss', test_loss/j, global_step=current_epo)

        return test_loss

    @staticmethod
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-7)
        return recall

    @staticmethod
    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-7)
        return precision

    @staticmethod
    def f1(y_true, y_pred):
        
        """ 
        
        Calculate F1-score.
        
        Parameters
        ----------
        y_true : 1D array
            Ground truth labels. 
            
        y_pred : 1D array
            Predicted labels.     
            
        Returns
        -------  
        f1 : float
            Calculated F1-score. 
            
        """     
        
        precision = EQTransformer_Transfer_Learning.precision(y_true, y_pred)
        recall = EQTransformer_Transfer_Learning.recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+1e-7))

    @staticmethod
    def check_accuracy(loader, model,current_epo, add_scal = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f1_score_det_batch = 0
        f1_score_p_batch = 0
        f1_score_s_batch = 0

        prec_det_batch = 0
        prec_p_batch = 0
        prec_s_batch = 0

        recall_det_batch = 0
        recall_p_batch = 0
        recall_s_batch = 0

        model.eval()
        j = 0
        with torch.no_grad():
            for batch in loader:

                # Compute prediction        
                det_pred, p_pred, s_pred = model(batch["X"].to(model.device))
                
                p_true = batch["y"][:, 0].to(model.device)
                s_true = batch["y"][:, 1].to(model.device)
                det_true = batch["detections"][:, 0].to(model.device)
        
                prec_det_batch += EQTransformer_Transfer_Learning.precision(det_true, det_pred)
                prec_p_batch += EQTransformer_Transfer_Learning.precision(p_true, p_pred)
                prec_s_batch += EQTransformer_Transfer_Learning.precision(s_true, s_pred)

                recall_det_batch += EQTransformer_Transfer_Learning.recall(det_true, det_pred)
                recall_p_batch += EQTransformer_Transfer_Learning.recall(p_true, p_pred)
                recall_s_batch += EQTransformer_Transfer_Learning.recall(s_true, s_pred)

                f1_score_det_batch += EQTransformer_Transfer_Learning.f1(det_true, det_pred)
                f1_score_p_batch += EQTransformer_Transfer_Learning.f1(p_true, p_pred)
                f1_score_s_batch += EQTransformer_Transfer_Learning.f1(s_true, s_pred)

                j += 1
        prec_det = prec_det_batch/j
        prec_p = prec_p_batch/j
        prec_s = prec_s_batch/j

        recall_det = recall_det_batch/j
        recall_p = recall_p_batch/j
        recall_s = recall_s_batch/j

        f1_score_det = f1_score_det_batch/j
        f1_score_p = f1_score_p_batch/j
        f1_score_s = f1_score_s_batch/j

        print(
            f" det f1_score: {f1_score_det*100:.3f}\n"
            f" det precision: {prec_det*100:.3f}\n"
            f" det recall: {recall_det*100:.3f}\n"
        )

        print(
            f" P f1_score: {f1_score_p*100:.3f}\n"
            f" P precision: {prec_p*100:.3f}\n"
            f" P recall: {recall_p*100:.3f}\n"
        )

        print(
            f" S f1_score: {f1_score_s*100:.3f}\n"
            f" S precision: {prec_s*100:.3f}\n"
            f" S recall: {recall_s*100:.3f}\n"
        )


        print(
            f" overall f1_score: {(f1_score_s+f1_score_p+f1_score_det)*33.333:.3f}\n"
        )

        if add_scal == True:
            writer.add_scalar('det f1 score', f1_score_det, global_step=current_epo)
            writer.add_scalar('P f1 score', f1_score_p, global_step=current_epo)
            writer.add_scalar('S f1 score', f1_score_s, global_step=current_epo)
            writer.add_scalar('overall f1 score', 0.333*(f1_score_s+f1_score_p+f1_score_det), global_step=current_epo)

            writer.add_scalar('det precision', prec_det, global_step=current_epo)
            writer.add_scalar('P precision', prec_p, global_step=current_epo)
            writer.add_scalar('S precision', prec_s, global_step=current_epo)

            writer.add_scalar('det recall', recall_det, global_step=current_epo)
            writer.add_scalar('P recall', recall_p, global_step=current_epo)
            writer.add_scalar('S recall', recall_s, global_step=current_epo)

        return f1_score_det, f1_score_p, f1_score_s
    
    def save_checkpoint(self, model, file_name='transfer_learing_EQT.pth.tar'):
        print('==> saving check_point')
        torch.save(model.state_dict(), file_name)

class SoftCrossEntropyLoss(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, y_hat, y):
      p = F.log_softmax(y_hat, 1)
      w_labels = y
      loss = -(w_labels*p).sum() / (w_labels).sum()
      return loss

class SoftCrossEntropyLossMean(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, eps=1e-7):
    
        h = y_true * torch.log(y_pred + eps)
        if y_pred.ndim == 3:
            h = h.mean(-1).sum(
                -1
            )  # Mean along sample dimension and sum along pick dimension
        else:
            h = h.sum(-1)  # Sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h
    
class Annotation(object):
    def __init__(self):
        
        # load model
        model = sbm.EQTransformer.from_pretrained('instance').to(device="cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load('transfer_learing_EQT.pth.tar')
        model.load_state_dict(checkpoint)
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

    with open('instance_eqtransformer.json', "r") as f:
        config = json.load(f)

    phasenet_obj = EQTransformer_Transfer_Learning(config)


    phasenet_obj()
    
    start_year_analysis = 2011
    start_day_analysis = 90
    end_year_analysis = 2011
    end_day_analysis = 90

    

    #data = sbd.ETHZ(sampling_rate=100)
    #train, dev, test = data.train_dev_test()


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
    '''
    '''
    base_path ='/home/javak/Transfer-Deep-Learning-chile-subduction-zone/transfer_learning/Creat_datasets/day1'

    base_model_list = ['instance', 'original']
    loss_weight_range = [2,10]
    batch_size_list = [128, 512]
    tun_obj = Parameters_Tuning (base_path, base_model_list, loss_weight_range, batch_size_list)
    tun_obj()
    '''
    

    