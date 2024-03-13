#optim=SGD; n_epochs=10; first_obj=top1
optim=SAM; n_epochs=10; first_obj=top1_robust
gpu=2
#
dataset=cifar10; res=32 #res1=32; res2=96
#dataset=cifar100; res=32; # res1=32; res2=96
#dataset=tinyimagenet; res=128; #res1=64; res2=128
#
#first_obj=robustness
#sec_obj=c_params; pmax=5.0
alpha=0.1
sigma=0.05
#sec_obj=macs # macs, params, activations, tinyml
#
iterations=28
resume_iter=0
#
seed=1

python cnas.py --n_gpus 1 --gpu 1 --gpu_list $gpu --n_workers 4 \
        --data ../datasets/$dataset --dataset $dataset \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --trainer_type entropic \
        --alpha $alpha --res $res --optim $optim \
        --first_obj $first_obj --first_predictor as \
        --sigma_min $sigma --sigma_max $sigma \
        --iterations $iterations --n_epochs $n_epochs --seed $seed \
        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma-ep$n_epochs-10jan \
        --resume ../results/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/iter_${resume_iter}
        #--sec_obj $sec_obj --pmax $pmax
        #--save ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma \
