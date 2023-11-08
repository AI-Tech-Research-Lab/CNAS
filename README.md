This repository contains the code for Constrained Neural Architecture Search (CNAS) and Adaptive Neural Architecture Search for Early Exit Neural Networks (EDANAS).

#References

## CNAS: Constrained Neural Architecture Search
Published in 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC)

Neural Architecture Search (NAS) paves the way for the automatic definition of neural networks architectures. The research interest in this field is steadily growing with several solutions available in the literature. This study introduces, for the first time in the literature, a NAS solution, called Constrained NAS (CNAS), able to take into account constraints on the search of the designed neural architecture. Specifically, CNAS is able to consider both functional constraints (i.e., the type of operations that can be carried out in the neural network) and technological constraints (i.e., constraints on the computational and memory demand of the designed neural network). CNAS has been successfully applied to Tiny Machine Learning and Privacy-Preserving Deep Learning with Homomorphic Encryption being two relevant and challenging application scenarios where functional and technological constraints are relevant in the neural network search.

Link to the paper: https://ieeexplore.ieee.org/document/9945080

## EDANAS: Adaptive Neural Architecture Search for Early Exit Neural Networks
Published in 2023 IEEE International Joint Conference on Neural Networks (IJCNN)

Early Exit Neural Networks (EENNs) endow neural network architectures with auxiliary classifiers to progressively process the input and make decisions at intermediate points of the network. This leads to significant benefits in terms of effectiveness and efficiency such as the reduction of the average inference time as well as the mitigation of overfitting and vanishing gradient phenomena. Currently, the design of EENNs, which is a very complex and time-consuming task, is carried out manually by experts. This is where Neural Architecture Search (NAS) comes into play by automatically designing neural network architectures focusing also on the optimization of the computational demand of these networks. These requirements are crucial in the design of machine and deep learning solutions meant to operate in devices constrained by the technology (computation, memory, and energy) such as Internet-Of-Things and embedded systems. Interestingly, few NAS solutions have taken into account the design of early exiting mechanisms. This work introduces, for the first time in the literature, a framework called Early exit aDAptive Neural Architecture Search (EDANAS) for the automatic design of both the EENN architecture and the parameters that manage its early exit mechanism in order to optimize both the accuracy in the classification tasks and the computational demand. EDANAS has proven to compete with expert-designed early exit solutions paving the way for a new era in the prominent field of NAS.

Link to the paper: https://ieeexplore.ieee.org/document/10191876

## Requirements

1. Install requirements with `pip install -r requirements.txt`

## Contents

- `scripts` contains the scripts needed to perform a search using NAS
- `results` contains the architecture found by search procedure (if any)
- `benchmarks` contains the architectures used for benchmarking (scroll down to see the benchmarks) found by search procedure. This folder is not created by the NAS procedure, it is only used to prove the results.

## Dataset
Download the dataset from the link embedded in the name.
| Dataset | Type | Train Size | Test Size | #Classes |
|:-:|:-:|:-:|:-:|:-:|
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)* |  | 50,000 | 10,000 | 10 |


## How to use CNAS to search
```python
""" Bi-objective search
Syntax: python msunas.py \
    --resume /path/to/iter/stats \ # iter stats file path
    --sec_obj [params/macs/activations/tiny_ml] \ # objective (in addition to top-1 acc)
    --n_gpus 8 \ # number of available gpus
    --gpu 1 \ # required number of gpus per evaluation job
    --n_workers 0 \ # number of threads for dataloader
    --n_epochs 5 \ # number of epochs to SGD training. 5 is the suggested value.
    --dataset [cifar10/cifar100/imagenet] \ # dataset 
    --n_classes [10/100/1000] \ # number of classes
    --data /path/to/data/ \ # dataset folder path
    --predictor [as] \ # type of accuracy predictor. 'as' stands for Adaptive Switching
    --supernet_path /path/to/supernet/weights \ # supernet model path
    --pretrained \ # flag to use the supernet pretrained weights
    --save /path/to/results/folder \ # results folder path
    --iterations [30] \ # number of NAS iterations
    --vld_size [10000/5000/...] \ # number of subset images from training set to guide search 
    # NOVEL PARAMETERS w.r.t. MSUNAS
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
- It supports searching for *params*, *macs*, *activations*, *tiny_ml* as the second objective. *Params*, *macs*, *activations* stand for the number of parameters, the number of macs, the sum of the activation sizes respectively.
- Choose an appropriate `--vld_size` to guide the search, e.g. 5,000 for CIFAR-10.
- To launch the search you can simply run a file in the `scripts` folder.
- Output file structure:
  - Every architecture sampled during search has `net_x.subnet` and `net_x.stats` stored in the corresponding iteration dir. 
  - A stats file is generated by the end of each iteration, `iter_x.stats`; it stores every architectures evaluated so far in `["archive"]`, and iteration-wise statistics, e.g. hypervolume in `["hv"]`, accuracy predictor related in `["surrogate"]`.
  - In case any architectures failed to evaluate during search, you may re-visit them in `failed` sub-dir under experiment dir. 
- You can resume the search from a specific previously performed iteration using the `resume` option and specifying the path to the `iter_x.stats` file
  
  
`tiny_ml` = 
$wp \times (params + penalty \times (max(0,params - pmax))) + wm \times (macs + penalty \times (max(0,macs - mmax)))$
$+ wa \times (activations + penalty \times (max(0,activations - amax)))$

## How to choose architectures
Once the search is completed, you can choose suitable architectures by:
- You have preferences, e.g. architectures with xx.x% top-1 acc. and xxxM FLOPs, etc.
```python
""" Find architectures with objectives close to your preferences
Syntax: python post_search.py \
    --save search-imagenet/final \ # path to the dir to store the selected architectures
    --expr search-imagenet/iter_30.stats \ # path to last iteration stats file in experiment dir
    -n 3 \ # number of desired architectures you want, the most accurate archecture will always be selected 
    --supernet_path /path/to/imagenet/supernet/weights \
    --prefer top1#80+flops#150 \ # your preferences, i.e. you want an architecture with 80% top-1 acc. and 150M FLOPs 
    --save_stats_csv \ # flag to be set whether to save post search results 
    --n_classes \ #number of classes 
    --pmax [2] \ # max number of params for candidate architecture
    --mmax [100] \ # max number of macs for candidate architecture
    --amax [5] \ # max number of activations for candidate architecture
    --wp [1] \ # weight for params
    --wm [1/40] \ # weight for macs
    --wa [1] \ # weight for activations
    --penalty [10**10] # penalty factor
"""
```
- If you do not have preferences, pass `None` to argument `--prefer`, architectures will then be selected based on trade-offs. 
- If you want the tiniest model according to the definitin of `tiny_ml` objective, pass `tiny_ml` to argument `--prefer`. Tiniest stands for the lowest `tiny_ml`   value.
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

## Benchmarks
- TinyML

| Model | NAS | Params | MACs | Act. Sizes | Accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|
| TinyFR (our) | yes | 2.89M | 126.17M | 5.21M | 97.2% |
| TinyRasp1% (our) | yes | 2.14M | 6.85M | 0.23M | 89.9% |
| MSUNAS | yes | 2.53M | 194.16M | 8.2M | 97.75% |
| MuxNet | yes | 2.1M | 200M |  | 98.0% |
| HCGNet-A1 | no | 1.1M | 100M |  | 96.85% |
| CCT-2/3x2 | no | 0.28M | 30M |  | 89.17% |
| CCT-7/3x1 | no | 3.76M | 950M |  | 98.0% |

TinyRasp1% experiment folders (tiny_ml): search-cifar10-mbv3-w1.0-2022-03-23 > final > net-tiny_ml@3 (NAS) + TinyMLRaspFineTuning (additional fine-tuning)

TinyFR experiment folders (tiny_ml): search-cifar10-mbv3-w1.0_07_03_2022 > final > net-tiny_ml@9 (NAS) + finetuningcifar10tiny192-2022_03_09 (additional fine-tuning)

MSUNAS experiment folders (noconstraints): search-noconstraintsFR-cifar10-mbv3-w1.0 > final > net-tiny_ml@3 (NAS) + finetuningPART1MSUNASFR + finetuningPART2MSUNASFR (additional fine-tuning)

- Privacy Preserving Deep Learning with Homomorphic Encryption

| Model | NAS | Params | MACs | Act. Sizes | Accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|
| CResNetHE (our) | yes | 0.49M | 137.51M | 1.19M | 74.72% |
| LeNet5| no | 0.14M | 1.5M | 0.018M | 46.0% |
| SingleLayerNN | no | 0.05M | 0.05M | 0.005M | 41.0% |

CResNetHE experiment folders (he): search-cifar10-resnet50_he > final > net-trade-off@5 (NAS)




