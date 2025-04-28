# Search an Early-Exit MobileNetV3 on CIFAR-10 optimizing on top1 accuracy and average number of MACs with a constraint on the maximum number of MACs and the
# minimum top1 accuracy. (NACHOS)

dataset=cifar10 val_split=0.1
seed=1

python cnas.py --sec_obj avg_macs --resume results/nachos_${dataset}_constraints_seed${seed}/iter_0 \
              --n_gpus 1 --gpu 1 --n_workers 4  --seed 1 \
              --data datasets/$dataset --dataset $dataset \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/nachos_${dataset}_constraints_seed${seed} --iterations 10 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --method bernulli --support_set --tune_epsilon\
              --val_split $val_split \
              --n_epochs 10 --warmup_ee_epochs 5 --ee_epochs 5 \
              --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 \
              --mmax 2.7 --top1min 0.65 \
              --lr 32 --ur 32 --rstep 4