# Search a MobileNetV3 on CIFAR-10 using SAM optimizer and optimizing on top1 robust and constrained params (FlatNAS)

optim=SGD; n_epochs=6; first_obj=top1
#optim=SAM; n_epochs=6; first_obj=top1_robust
gpu=0
#
dataset=cifar10
#dataset=cifar100
#dataset=tinyimagenet
#
#first_obj=robustness
sec_obj=tiny_ml
pmax=5.0
alpha=0.5
sigma=0.05
#sec_obj=macs # macs, params, activations, tinyml
#
iterations=30
resume_iter=1
#
seed=1

lr=128 #min resolution
ur=224 #max resolution
rstep=4 #resolution step

python cnas.py --resume ../results/entropic-mbv3-$dataset-$optim-$first_obj-c_params-max$pmax-alpha$alpha-sigma$sigma-ep$n_epochs-multires-balance/iter_3 --n_gpus 1 --gpu 1 --gpu_list $gpu --n_workers 4 \
        --data ../datasets/$dataset --dataset $dataset \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --trainer_type single_exit \
        --alpha $alpha --lr $lr --ur $ur --rstep $rstep --optim $optim \
        --first_obj $first_obj --first_predictor gp \
        --sigma_min $sigma --sigma_max $sigma \
        --iterations $iterations --n_epochs $n_epochs --seed $seed \
        --sec_obj $sec_obj --pmax $pmax --wp 1.0 \
        --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-c_params-max$pmax-alpha$alpha-sigma$sigma-ep$n_epochs-multires-balance