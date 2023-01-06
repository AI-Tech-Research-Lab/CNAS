python msunas.py --sec_obj tiny_ml --n_doe 1 \
              --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
              --dataset cifar10 --n_classes 10 \
              --data ../data/cifar10 \
              --predictor as --supernet_path ./ofa_nets/ofa_eembv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/cifar10-mbv3-adaptive --iterations 10 --vld_size 5000 \
              --we 0.4 \
              --lr 40 --ur 104
