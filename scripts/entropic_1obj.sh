optim=SGD; n_epochs=6; first_obj=top1
#optim=SAM; n_epochs=6; first_obj=top1_robust
gpu=0
#
#dataset=cifar10; res=32
dataset=cifar100; res=32
#dataset=tinyimagenet; res=128
#
#first_obj=robustness
#first_obj=top1_robust
alpha=0.9
sigma=0.025
#
iterations=12
#resume_iter=28
#
seed=1

python msunas.py --resume ../results/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma/iter_18 --n_gpus 1 --gpu 1 --n_workers 4 \
        --data ../datasets/$dataset --dataset $dataset --optim $optim \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --trainer_type entropic \
        --first_obj $first_obj --first_predictor as \
        --sigma_min $sigma --sigma_max $sigma --alpha $alpha --res $res\
        --iterations $iterations --n_epochs $n_epochs --seed $seed \
        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma
#        --resume ../results/entropic-mbv3-$dataset-$optim-$first_obj/iter_${resume_iter}
