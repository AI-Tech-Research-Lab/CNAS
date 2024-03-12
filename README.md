This repository contains the code for a Neural Architecture Search framework, based on MSUNAS[1], supporting for:
- technological and functional constraints (introduced in CNAS[2])
- Early Exit classifiers on top of OFA[3] backbones (introduced in EDANAS[4]) and constrained in their number of MAC operations (introduced in NACHOS)
- Out-Of-Distribution robustness optimization accounting for the flatness of the loss landscape (introduced in FLATNAS)

## Requirements

1. Install requirements with `pip install -r requirements.txt`

## Contents

- `scripts` contains the scripts needed to perform a search using NAS.
- `results` contains the architecture found by search procedure (if any).
- `supernets` contains the pretrained weights of the OFA supernets. Currently, we provide only the OFA MobilenetV3-based supernet.
- `acc_predictor` contains the accuracy predictors used to evaluate candidate networks in the NAS search.
- `early_exit` contains the trainer and the utilities for early exit
- `robustness` contains the trainer and the utilities for robustness optimization

## Dataset
Download the dataset from the link embedded in the name.
| Dataset | Type | Train Size | Test Size | #Classes |
|:-:|:-:|:-:|:-:|:-:|
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)* |  | 50,000 | 10,000 | 10 |

## How to launch the search
```python
Syntax: python msunas.py \
    --resume /path/to/iter/stats \ # iter stats file path
    --sec_obj [params/macs/activations/tiny_ml/avg_macs] \ # objective (in addition to top-1 acc)
    --n_gpus 8 \ # number of available gpus
    --gpu 1 \ # required number of gpus per evaluation job
    --n_workers 0 \ # number of threads for dataloader
    --n_epochs 5 \ # number of epochs to SGD training. 5 is the suggested value.
    --dataset [cifar10/cifar100/imagenet] \ # dataset 
    --n_classes [10/100/1000] \ # number of classes
    --data /path/to/data/ \ # dataset folder path
    --predictor [as/carts/rbf/mlp/gp] \ # type of accuracy predictor. 'as' stands for Adaptive Switching
    --supernet_path /path/to/supernet/weights \ # supernet model path
    --search_space \ # ['mobilenetv3', 'eemobilenetv3']
    --pretrained \ # flag to use the supernet pretrained weights
    --save /path/to/results/folder \ # results folder path
    --iterations [30] \ # number of NAS iterations
    --vld_size [10000/5000/...] \ # number of subset images from training set to guide search 
    --pmax [2] \ # max number of params for candidate architecture
    --mmax [100] \ # max number of macs for candidate architecture
    --amax [5] \ # max number of activations for candidate architecture
    --wp [1] \ # weight for params
    --wm [1/40] \ # weight for macs
    --wa [1] \ # weight for activations
    --penalty [10**10] # penalty factor
    --lr [192] \ # minimum resolution of the search
    --ur [256] # maximum resolution of the search
"""
```
- To launch the search, you can use the above command with the proper syntax or simply run a file in the `scripts` folder.
- It supports searching for *params*, *macs*, *activations*, *tiny_ml*, *avg_macs* as the second objective. *params*, *macs*, *activations* stand for the number of parameters, the number of macs, the sum of the activation sizes respectively.
- *tiny_ml* is a figure of merit, introduced in CBNAS, accounting jointly for *params*, *macs*, *activations* and the related constraints.
- *avg_macs* is a figure of merit, introduced in EDANAS, for Early Exit Neural Networks accounting for the sum of the number of MACs of each exit of an EENN weighted by the exit ratios of the data. It is different from *macs* which accounts for the number of MACs of the backbone network (i.e., considering the network with a single exit, the final classifier).
- Choose an appropriate `--vld_size` to guide the search, e.g. 5,000 for CIFAR-10.
- To launch the search you can simply run a file in the `scripts` folder.
- Output file structure:
  - Every architecture sampled during search has `net_x.subnet` and `net_x.stats` stored in the corresponding iteration dir. 
  - A stats file is generated by the end of each iteration, `iter_x.stats`; it stores every architectures evaluated so far in `["archive"]`, and iteration-wise statistics, e.g. hypervolume in `["hv"]`, accuracy predictor related in `["surrogate"]`.
  - In case any architectures failed to evaluate during search, you may re-visit them in `failed` sub-dir under experiment dir. 
- You can resume the search from a specific previously performed iteration using the `resume` option and specifying the path to the `iter_x.stats` file.
- `--search_space` is used to select whether to search single-exit MobileNets (`mobilenetv3`) or multi-exits (i.e., Early Exit Neural Networks) Mobilenets (`eemobilenetv3`).

## How to choose architectures
Once the search is completed, you can choose suitable architectures by:
- You have preferences, e.g. architectures with xx.x% top-1 acc. and xxxM FLOPs, etc.
```python
""" Find architectures with objectives close to your preferences
Syntax: python post_search.py \
    --save search-cifar10/final \ # path to the dir to store the selected architectures
    --expr search-cifar10/iter_30.stats \ # path to last iteration stats file in experiment dir
    -n 3 \ # number of desired architectures you want, the most accurate archecture will always be selected 
    --supernet_path /path/to/OFA/supernet/weights \
    --prefer params \ # your preferences, i.e. you want an architecture with the lowest sec_obj value (e.g., 'params') or alternatively with the best trade-off ('trade-off')
    --n_exits \ # optional, used to filter the results by the number of exits of the network
    --save_stats_csv \ # flag to be set whether to save post search results 
"""
```
- If you do not have preferences, pass `trade-off` to argument `--prefer`, architectures will then be selected based on trade-offs. 
- All selected architectures should have three files created:
  - `net.subnet`: use to sample the architecture from the supernet
  - `net.config`: configuration file that defines the full architectural components
  - `net.inherited`: the inherited weights from supernet
  
## How to validate architectures
To realize the full potential of the searched architectures, we further fine-tune from the inherited weights. Assuming that you have both `net.config` and `net.inherited` files. 

```python
""" Fine-tune on CIFAR-10 from inherited weights
Syntax: python train_cifar.py \
    --data /path/to/CIFAR-10/data/ \
    --model [nsganetv2_s/nsganetv2_m/...] \ # just for naming the output dir
    --model-config /path/to/model/.config/file \
    --img-size [192, ..., 224, ..., 256] \ # image resolution, check "r" in net.subnet
    --drop 0.2 --drop-path 0.2 \
    --cutout --autoaugment --save
"""
```

## References

[1] NSGANetV2: Evolutionary Multi-Objective Surrogate-Assisted Neural Architecture Search, ECCV 2020 (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460035.pdf)

[2] CNAS: Constrained Neural Architecture Search, SMC 2022 (https://ieeexplore.ieee.org/document/9945080)

[3] ONCE-FOR-ALL: TRAIN ONE NETWORK AND SPECIALIZE IT FOR EFFICIENT DEPLOYMENT, ICLR 2019 (https://openreview.net/pdf?id=HylxE1HKwS)

[4] EDANAS: Adaptive Neural Architecture Search for Early Exit Neural Networks, IJCNN 2023 (https://ieeexplore.ieee.org/document/10191876)

[5] NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (https://arxiv.org/abs/2401.13330)

[6] FlatNAS: optimizing Flatness in Neural Architecture Search for Out-of-Distribution Robustness (https://arxiv.org/abs/2402.19102)
