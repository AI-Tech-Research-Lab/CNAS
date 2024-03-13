optim=SGD
#optim=SAM
dataset=cifar10
seed=1
first_obj=top1
#first_obj=robustness
sec_obj=params

python cnas.py --n_gpus 1 --gpu 1 --n_workers 4 \
        --data ../datasets/$dataset --dataset $dataset \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --trainer_type entropic \
        --lr 32 --ur 32 --rstep 1 --optim $optim \
        --first_obj $first_obj --first_predictor mlp \
        --sigma_min 0.05 --sigma_max 0.06 --sigma_step 0.01 \
        --iterations 30 --n_epochs 10 --seed $seed \
        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-adaptivesigma \
#	--resume ../results/entropic-mbv3-$dataset-$optim-$first_obj/iter_27

#        --sec_obj $sec_obj \
#        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj \

