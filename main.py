# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from torchsummary import summary

import numpy as np

import os
from utils import metrics, sample_gt, show_results
from datasets import get_dataset, HyperX, DATASETS_CONFIG
from models.get_model import get_model
from train import train, test

import argparse
import torch.optim as optim

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SSRN, FDSSC, DFFN, DHCNet, BASSNet, SPRN"
                    "HPDM-SSRN, HPDM-FDSSC, HPDM-DFFN, HPDM-DHCNet, HPDM-BASSNet, HPDM-SPRN"
                    )
parser.add_argument('--folder', type=str, help="Folder where to store the datasets.",
                    default="./Datasets/")
parser.add_argument('--runs', type=int, default=1, 
                    help="Number of runs (default: 1)")
parser.add_argument('--patch_size', type=int, default=7, 
                    help="Input patch size")
parser.add_argument('--percentage', type=float, default=0.01, 
                    help="Training percentage (include validation samples, 50% training samples are used for validation)")
parser.add_argument('--data_aug', action='store_true', 
                    help="If use data augmentation")

parser.add_argument('--cuda', type=str, default='-1',
                    help="Specify CUDA device")

args = parser.parse_args()

if int(args.cuda) < 0:
    print("Computation on CPU")
    device = torch.device('cpu')
elif torch.cuda.is_available():
    print("Computation on CUDA GPU device {}".format(args.cuda))
    device = torch.device('cuda:{}'.format(args.cuda))
else:
    print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
    device = torch.device('cpu')

# Dataset name
DATASET = args.dataset
# Training percentage
PERCENTAGE = args.percentage
# Data augmentation
DATA_AUG = args.data_aug
# Model name
MODEL = args.model
# Number of runs 
N_RUNS = args.runs
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Patch size
PATCH_SIZE = args.patch_size
# Training epoch
EPOCH = 300
#Initial learning rate
LR = 1e-3
#Batch size
BATCH_SIZE = 64

#print parameters
print('Dataset:%s' % args.dataset)
print('patch size:%d' % PATCH_SIZE)
print('epoch:%d;lr:%f;batch_size:%d' % (EPOCH, LR, BATCH_SIZE))
print('data augment:%s' % str(DATA_AUG))

# Load the dataset
img, gt, LABEL_VALUES = get_dataset(DATASET,FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
# Random seeds
seeds = [7,17,27,37,47,57,67,77,87,97]
# seeds = [7] * 10    #used for finding the best number of groups

results = []
# run the experiment several times
for run in range(N_RUNS):
    np.random.seed(seeds[run])
    # Sample random training spectra
    train_gt, test_gt = sample_gt(gt, PERCENTAGE, seed=seeds[run])
    # Split train set in train/val
    train_gt, val_gt = sample_gt(train_gt, 0.5, seed=seeds[run])
    print("Training samples: {}, validating samples: {}, total samples: {})".format(
        np.sum(train_gt>-1),np.sum(val_gt>-1), np.sum(gt>-1)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    # data
    train_dataset = HyperX(img, train_gt, patch_size=PATCH_SIZE, data_aug=DATA_AUG)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    drop_last=False,
                                    shuffle=True)

    val_dataset = HyperX(img, val_gt, patch_size=PATCH_SIZE)
    val_loader = data.DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                drop_last=False)

    #load model and loss
    model = get_model(MODEL, args.dataset, N_CLASSES, N_BANDS, PATCH_SIZE)
    if run == 0:
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            summary(model, input.size()[1:], device='cpu')
    model.to(device)
    #training config
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100,250])
    criterion = torch.nn.CrossEntropyLoss()
    model_dir = './checkpoints/' + MODEL + "/" + args.dataset + "/" + str(run)

    try:
        train(model, optimizer, criterion, train_loader, val_loader, EPOCH, model_dir, scheduler=scheduler, device=device)
    except KeyboardInterrupt:
        # Allow the user to stop the training
        pass
    
    #testing model
    probabilities = test(model, model_dir, img, PATCH_SIZE, N_CLASSES, device=device)
    prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(prediction, test_gt, n_classes=N_CLASSES)

    results.append(run_results)
    show_results(run_results, label_values=LABEL_VALUES)
    del model, train_dataset, train_loader, val_dataset, val_loader

if N_RUNS > 1:
    show_results(results, label_values=LABEL_VALUES, agregated=True)

