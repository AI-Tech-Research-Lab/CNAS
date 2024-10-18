# minimum requirements to run cnas with multiexit --> create a subnet & inserted in results/ folder

python cnas.py --sec_obj tiny_ml \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data datasets/cifar10 --dataset cifar10 \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/search_path --iterations 0 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --method bernulli --support_set --tune_epsilon\
              --val_split 0.1 \
              --n_epochs 0 --warmup_ee_epochs 0 --ee_epochs 0 \
              --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 \
              --mmax 2.7 --top1min 0.65 \
              --lr 32 --ur 32 --rstep 4 \
              --n_doe 10