optim=SGD
#optim=SAM
dataset=cifar10
seed=1
first_obj=top1
#first_obj=robustness

python cnas.py --gpu 1 --gpu_list 0 2 \
        --n_workers 4 \
        --first_obj robustness --first_predictor mlp \
        --data ../datasets/$dataset --dataset $dataset \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --seed $seed \
        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-adaptivesigma-prova\
        --iterations 30 --n_epochs 10 --trainer_type entropic \
        --lr 32 --ur 32 --rstep 1 --optim $optim \
        --sigma_min 0.05 --sigma_max 0.06 --sigma_step 0.01   
#       --sec_obj params # togliendolo fa un solo obiettivo
#sigma 0.05
