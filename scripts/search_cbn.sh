python msunas.py --resume ../results/cifar10-cbn-mbv3-8sept-nocalibration/iter_2 --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --dataset cifar10 \
              --predictor as --supernet_path ./ofa_nets/ofa_cbnmbv3 --pretrained  \
              --save ../results/cifar10-cbn-mbv3-8sept-nocalibration --iterations 10 \
              --trainer_type multi_exits \
              --fmax 2.7 --top1max 0.65 \
              --lr 32 --ur 32 

