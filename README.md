# Spectral Partitioning Residual Network with Spatial Attention Mechanism for Hyperspectral Image Classification
This repository is the implementation of our paper: [Spectral Partitioning Residual Network with Spatial Attention Mechanism for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/9454961). 

If you find this work helpful, please cite our paper:

    @ARTICLE{9454961,  
    author={Zhang, Xiangrong and Shang, Shouwang and Tang, Xu and Feng, Jie and Jiao, Licheng},  
    journal={IEEE Transactions on Geoscience and Remote Sensing},   
    title={Spectral Partitioning Residual Network With Spatial Attention Mechanism for Hyperspectral Image Classification},   
    year={2021},  
    volume={},  number={},  
    pages={1-14},  
    doi={10.1109/TGRS.2021.3074196}}

## Requirements
Only Python3 is supported. We recommend you to create a Python virtual environment and then run the following command to install dependencies.

    pip install -r requirement.txt

CUDA and cuDNN are optional

## Datasets
You can download hyperspectral image datasets at <http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes>, and move the files to `./Datasets` folder

## Usage
To train a model, simply run `main.py`, for example:

    python main.py --dataset PaviaU --model HPDM-SPRN --runs 10  --patch_size 7 --percentage 0.01 --data_aug

To get colored results, run `eval.py`. The colored results can be found in the `results` folder. For example:
    
    python eval.py --dataset PaviaU --model HPDM-SPRN --patch_size 7 --weights (saved model path)

## Models
- [DFFN](https://ieeexplore.ieee.org/document/8283837) (HPDM-DFFN)
- [DHCNet](https://ieeexplore.ieee.org/document/8361481) (HPDM-DHCNet)
- [BASSNet](https://ieeexplore.ieee.org/document/7938656) (HPDM-BASSNet)
- [SSRN](https://ieeexplore.ieee.org/document/8061020) (HPDM-SSRN)
- [FDSSC](https://www.mdpi.com/2072-4292/10/7/1068/htm) (HPDM-FDSSC)
- SPRN (HPDM-SPRN)

## Acknowledgement
Part of our codes references to the project [DeepHyperX](https://github.com/nshaud/DeepHyperX). 

