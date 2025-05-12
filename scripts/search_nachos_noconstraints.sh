#!/bin/sh

# w_beta = 0.0 and no --mmax and --top1min

datasets="cinic10 imagenette"
seeds="1 2 3 4"
val_split=0.1

for dataset in $datasets; do
  for seed in $seeds; do
    echo "Launching search for dataset=$dataset, seed=$seed..."

    # Default values
    lr=32
    ur=32

    # Special case for imagenette
    if [ "$dataset" = "imagenette" ]; then
      lr=160
      ur=160
    fi

    python cnas.py --sec_obj avg_macs \
                   --n_gpus 1 --gpu 1 --n_workers 4 --seed $seed \
                   --data datasets/$dataset --dataset $dataset \
                   --first_predictor as --sec_predictor as \
                   --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained \
                   --save results/nachos_${dataset}_noconstraints_seed${seed} --iterations 10 \
                   --search_space cbnmobilenetv3 --trainer_type multi_exits \
                   --method bernulli --support_set --tune_epsilon \
                   --val_split $val_split \
                   --n_epochs 10 --warmup_ee_epochs 5 --ee_epochs 5 \
                   --w_alpha 1.0 --w_beta 0.0 --w_gamma 1.0 \
                   --lr $lr --ur $ur --rstep 4
  done
done

