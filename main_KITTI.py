import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

from multiprocessing import cpu_count

base_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = "H:/KITTI/RAW/total"
# test a given network
# address = os.path.join(base_dir, 'results/KITTI/2022_05_20_08_29_38/')
# or test the last trained network
address = "last"
################################################################################
# Network parameters
################################################################################
net_class = sn.GyroNet
net_params = {
    'in_dim': 6,
    'out_dim': 6+6+2,
    'c0': 16,
    'dropout': 0.2,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
}
################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.KITTIDataset
dataset_params = {
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    'predata_dir': os.path.join(base_dir, 'data/KITTI'),
    # set train, val and test sequence
    'train_seqs': [
         '2011_09_26_drive_0022_extract',
         '2011_09_29_drive_0071_extract',
         '2011_09_30_drive_0018_extract',
         '2011_09_30_drive_0020_extract',
         '2011_09_30_drive_0027_extract',
         '2011_09_30_drive_0028_extract',
         '2011_10_03_drive_0027_extract',
         '2011_10_03_drive_0034_extract',
         '2011_10_03_drive_0047_extract'
    ],
    'val_seqs': [
        '2011_09_26_drive_0022_extract',
        '2011_09_29_drive_0071_extract',
        '2011_09_30_drive_0018_extract',
        '2011_09_30_drive_0020_extract',
        '2011_09_30_drive_0027_extract',
        '2011_09_30_drive_0028_extract',
        '2011_10_03_drive_0027_extract',
        '2011_10_03_drive_0034_extract',
        '2011_10_03_drive_0047_extract'
        ],
    'test_seqs': [
        '2011_09_26_drive_0036_extract',
        '2011_09_26_drive_0101_extract',
        '2011_09_30_drive_0033_extract',
        '2011_09_30_drive_0034_extract',
        '2011_10_03_drive_0042_extract',


        '2011_09_26_drive_0022_extract',
        # '2011_09_26_drive_0036_extract',
        '2011_09_29_drive_0071_extract',
        '2011_09_30_drive_0018_extract',
        '2011_09_30_drive_0020_extract',
        '2011_09_30_drive_0027_extract',
        '2011_09_30_drive_0028_extract',
        '2011_10_03_drive_0027_extract',
        '2011_10_03_drive_0034_extract',
        '2011_10_03_drive_0047_extract'
        ],
    # size of trajectory during training
    'N': 40 * 100,

}
################################################################################
# Training parameters
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {

        'w':  1e6,
        'target': "all",
        'huber': 0.004,
        'dt': 0.01,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 100,
        'T_mult': 2,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 14,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 50,
    # total number of epochs
    'n_epochs': 1800,
    # where record results ?
    'res_dir': os.path.join(base_dir, "results/KITTI"),
    # where record Tensorboard log ?
    'tb_dir': os.path.join(base_dir, "results/runs/KITTI"),
}




################################################################################
# Train on training data set
################################################################################
#
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
   train_params['tb_dir'], net_class, net_params, None,
   train_params['loss']['dt'])
learning_process.train(dataset_class, dataset_params, train_params)

print("finish training")

################################################################################
# Test on full data set
################################################################################
learning_process = lr.GyroLearningBasedProcessing(train_params['res_dir'],
    train_params['tb_dir'], net_class, net_params, address=address,
    dt=train_params['loss']['dt'])
learning_process.test(dataset_class, dataset_params, ['test'])
print("finish testing")
