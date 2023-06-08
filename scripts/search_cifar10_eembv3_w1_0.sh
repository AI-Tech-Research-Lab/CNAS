python msunas.py --resume ../results/edanas-r32/iter_0 --sec_obj macs \
              --n_gpus 1 --gpu 1 --n_workers 4 --n_epochs 5 \
              --dataset cifar10 --n_classes 10 \
              --data ../data/cifar10 \
              --predictor mlp --supernet_path ./ofa_nets/ofa_eembv3 --pretrained  \
              --save ../results/edanas-r32 --iterations 30 --vld_size 5000 \
              --lr 32 --ur 32
