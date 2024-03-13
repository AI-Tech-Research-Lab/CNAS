dataset=cifar10; res=32; ood_data="../datasets/cifar10c"
dataset=cifar100; res=32; ood_data="../datasets/cifar100c"
#dataset=tinyimagenet; res=64
#
optim=SGD; folder_optim=SGD; epochs_optim=10; first_obj=top1
optim=SAM; folder_optim=SAM; epochs_optim=10; first_obj=top1_robust
epochs=10
#
alpha=0.9
sigma=0.05
#
seed=1
#
device=2

python robustness/train.py --dataset $dataset \
    --data ../datasets/$dataset --ood_data $ood_data --model mobilenetv3 --device $device \
    --model_path ../results/risultati-res32/entropic-mbv3-$dataset-${folder_optim}-$first_obj-alpha$alpha-sigma$sigma-ep$epochs_optim-10jan/final/net-${first_obj}_0/net.subnet \
    --output_path ../results/risultati-res32/entropic-mbv3-$dataset-${folder_optim}-$first_obj-alpha$alpha-sigma$sigma-ep$epochs_optim-10jan/final/net-${first_obj}_0 \
    --pretrained --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --res $res --epochs $epochs --optim $optim --save_ckpt --ood_eval --alpha $alpha \
    --eval_robust