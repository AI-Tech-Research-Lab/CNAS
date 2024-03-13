optim=SGD; n_epochs=10
optim=SAM; n_epochs=10
#
dataset=cifar10; res1=32; res2=96
#dataset=cifar100; res1=32; res2=96
#dataset=tinyimagenet; res1=64; res2=128
#
first_obj=top1
#first_obj=robustness
sec_obj=robustness
#sec_obj=macs # macs, params, activations, tinyml
#
iterations=30
#resume_iter=28
#
seed=1

python cnas.py --n_gpus 1 --gpu 1 --gpu_list 0 --n_workers 4 \
        --data ../datasets/$dataset --dataset $dataset \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --trainer_type entropic \
        --lr $res1 --ur $res2 --rstep 4 --optim $optim \
        --first_obj $first_obj --first_predictor as --sec_predictor as \
        --sigma_min 0.05 --sigma_max 0.05 \
        --iterations $iterations --n_epochs $n_epochs --seed $seed \
        --sec_obj $sec_obj \
        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj
 	
#        --resume ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj/iter_24
#        --resume ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj/iter_${resume_iter}

