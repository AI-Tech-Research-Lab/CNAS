python cnas.py  --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --dataset cifar100 \
              --predictor as --search_space cbnmobilenetv3 \
              --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/cifar100-cbn-mbv3-20oct --iterations 10 \
              --trainer_type multi_exits \
              --fmax 3.7 --top1min 0.35 \
              --lr 32 --ur 32 