#!/usr/bin/env bash

DEVICE=$1

python main.py +dataset=cifar10 +method=bernulli_logits  experiment=base +model=resnet20 optimizer=sgd +training=cifar10 hydra.run.dir='./outputs/ablation_logits/cifar10/resnet20/nosample_nodropout' training.device="$DEVICE"   experiment.load=true method.sample=false method.dropout=0 method.temperature_scaling=false
python main.py +dataset=cifar10 +method=bernulli_logits  experiment=base +model=resnet20 optimizer=sgd +training=cifar10 hydra.run.dir='./outputs/ablation_logits/cifar10/resnet20/noreg_nosample' training.device="$DEVICE"   experiment.load=true method.sample=false method.dropout=0 method.temperature_scaling=false method.beta=0
python main.py +dataset=cifar10 +method=bernulli_logits  experiment=base +model=resnet20 optimizer=sgd +training=cifar10 hydra.run.dir='./outputs/ablation_logits/cifar10/resnet20/noreg_sample' training.device="$DEVICE"   experiment.load=true method.beta=0

python main.py +dataset=cifar10 +method=bernulli_logits  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/ablation_logits/cifar10/resnet20/nosample_dropout'\
 training.device="$DEVICE"   experiment.load=true  \
  method.sample=false method.dropout=0.25 method.prior_mode=ones training.epochs=20

python main.py +dataset=cifar10 +method=bernulli_logits  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/ablation_logits/cifar10/resnet20/sample_nodropout' training.device="$DEVICE"   experiment.load=true method.sample=true method.dropout=0

python main.py +dataset=cifar10 +method=bernulli_logits  experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/ablation_logits/cifar10/resnet20/nosample_nodropout' training.device="$DEVICE"   experiment.load=true  method.sample=false method.dropout=0 method.prior_mode=ones training.epochs=20
