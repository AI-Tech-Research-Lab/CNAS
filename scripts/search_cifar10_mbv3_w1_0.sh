python msunas.py --sec_obj tiny_ml \
              --n_gpus 8 --gpu 1 --n_workers 4 --n_epochs 5 \
              --dataset cifar10 --n_classes 10 \
              --data ../data/cifar10 \
              --predictor as --supernet_path ./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/search-cifar10-mbv3-w1.0 --iterations 10 --vld_size 5000 \
              --pmax 2.2 --fmax 7 --amax 0.3 --wp 1 --wf 1 --wa 1 --penalty 10000000000 \
              --lr 40 --ur 104
