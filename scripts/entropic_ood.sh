optim=SGD
#optim=SAM
#
dataset=cifar10; res=128
#dataset=cifar100; res=32
#dataset=tinyimagenet; res=64
#
ood_data="../datasets/cifar10c"
#
model=mobilenetv3
#
first_obj=top1
#first_obj=robustness
#
epochs=1
#resume_iter=28
#
seed=1
#
device=0
alpha=0.5
sigma=0.05
n_epochs=10



python robustness/train.py --dataset $dataset --ood_data $ood_data --ood_eval \
    --data ../datasets/$dataset --model $model --device $device \
    --model_path ../results/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/final/net-top1_0/net.subnet \
    --output_path ../results/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/final/net-top1_0 \
    --pretrained --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --epochs 5 --optim $optim --save_ckpt --res 32 --load_ood


