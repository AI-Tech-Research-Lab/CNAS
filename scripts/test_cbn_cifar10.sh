python cnas.py --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 \
              --first_predictor as --sec_predictor as \
              --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/nachos-cifar10 --iterations 2 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --warmup_ee_epochs 2 --ee_epochs 3 \
              --mmax 2.7 --top1min 0.65 \
              --lr 32 --ur 32 