# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data

import numpy as np
import seaborn as sns
import imageio

import os
from utils import metrics, show_results
from datasets import get_dataset, DATASETS_CONFIG
from models.get_model import get_model
from train import test

import argparse

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
parser.add_argument('--patch_size', type=int, default=7, 
                    help="Input patch size")
parser.add_argument('--cuda', type=str, default='-1',
                    help="Specify CUDA device")
parser.add_argument('--weights', type=str, default=None,
                    help="Folder to the weights used for evaluation")
parser.add_argument('--output', type=str, default='./results',
                    help="Folder to store results")

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
# Model name
MODEL = args.model
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Weights to evaluate
WEIGHTS = args.weights
# Folder to restore results
OUTPUT = args.output
# Patch size
PATCH_SIZE = args.patch_size
#Batch size
BATCH_SIZE = 64

print('Dataset: %s' % DATASET)
print('patch size: %d' % PATCH_SIZE)
print('Model: %s' % (MODEL))

# Load the dataset
img, gt, LABEL_VALUES = get_dataset(DATASET,FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Generate color palette
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", N_CLASSES+1)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d

#load model and weights
model = get_model(MODEL, args.dataset, N_CLASSES, N_BANDS, PATCH_SIZE)
print('Loading weights from %s' % WEIGHTS + '/model_best.pth')
model = model.to(device)
model.load_state_dict(torch.load(WEIGHTS + '/model_best.pth'))
model.eval()

#testing model
probabilities = test(model, WEIGHTS, img, PATCH_SIZE, N_CLASSES, device=device)
prediction = np.argmax(probabilities, axis=-1)

run_results = metrics(prediction, gt, n_classes=N_CLASSES)

prediction[gt < 0] = -1

#color results
colored_gt = color_results(gt+1, palette)
colored_pred = color_results(prediction+1, palette)

outfile = os.path.join(OUTPUT, DATASET, MODEL)
os.makedirs(outfile, exist_ok=True)

imageio.imsave(os.path.join(outfile, DATASET+'_gt.png'), colored_gt)
imageio.imsave(os.path.join(outfile, DATASET+'_'+MODEL+'_out.png'), colored_pred)

show_results(run_results, label_values=LABEL_VALUES)
del model


