python msunas.py --resume ../results/cifar10-mbv3-adaptive/iter_19 --sec_obj macs \
              --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
              --dataset cifar10 --n_classes 10 \
              --data ../data/cifar10 \
              --predictor as --supernet_path ./ofa_nets/ofa_eembv3 --pretrained  \
              --save ../results/cifar10-mbv3-adaptive --iterations 1 --vld_size 5000 \
              --lr 40 --ur 104
