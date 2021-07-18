# -*- coding: utf-8 -*-
import numpy as np
import argparse
from utils import sample_gt
from datasets import DATASETS_CONFIG, get_dataset

if __name__ == '__main__':
    
    dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]
                     
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                        help="Dataset to use.") 
    parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="../Datasets/")
    parser.add_argument('--training_sample', type=float, default=10,
                        help="Percentage of samples to use for training (default: 10%%)")
    parser.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
    
    args = parser.parse_args()
    
    img, gt, _ = get_dataset(args.dataset, args.folder)
    train_gt, test_gt = sample_gt(gt, args.training_sample, mode=args.sampling_mode)
    train_gt, val_gt = sample_gt(train_gt, 0.5)
    #save file
    train_gt_path = args.folder + '/' + args.dataset + '/train_gt.npy'
    val_gt_path = args.folder + '/' + args.dataset + '/val_gt.npy'
    test_gt_path = args.folder + '/' + args.dataset + '/test_gt.npy'
    np.save(train_gt_path, train_gt)
    np.save(val_gt_path, val_gt)
    np.save(test_gt_path, test_gt)
    print("Done!")


