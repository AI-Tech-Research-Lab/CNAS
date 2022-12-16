#!/usr/bin/env bash

cd ../nsganetv2

python msunas.py   --resume ../results/search-cifar10-resnet50_he/iter_0 --sec_obj tiny_ml \
              --n_gpus 1 --gpu 1 --n_workers 4 --n_epochs 5 \
              --dataset cifar10HE --n_classes 10 \
              --data ../data/cifar10 \
              --predictor carts --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 \
              --save ../results/search-cifar10-resnet50 --iterations 10 --vld_size 5000 \
              --pmax 0.5 --fmax 150 --amax 5 --wp 1 --wf 0.00333 --wa 0.1 --penalty 10000000000 \
              --lr 40 --ur 50 
