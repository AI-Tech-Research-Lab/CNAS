# Search an Early-Exit MobileNetV3 on CIFAR-10 optimizing on top1 accuracy and average number of MACs with a constraint on the maximum number of MACs and the
# minimum top1 accuracy. (NACHOS)

python cnas.py --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4  --seed 1 \
              --data datasets/cifar10 --dataset cifar10 \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/nachos_cifar10_constraints_seed1 --iterations 10 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --method bernulli --support_set --tune_epsilon\
              --val_split 0.1 \
              --n_epochs 10 --warmup_ee_epochs 5 --ee_epochs 5 \
              --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 \
              --mmax 2.7 --top1min 0.65 \
              --lr 32 --ur 32 --rstep 4