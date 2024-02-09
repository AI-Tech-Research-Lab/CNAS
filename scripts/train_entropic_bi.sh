dataset=cifar10; res=208; ood_data="../datasets/cifar10c"
dataset=cifar100; res=224; ood_data="../datasets/cifar100c"
#dataset=tinyimagenet; res=64
#
device=1
#
optim=SGD; folder_optim=SGD; epochs_optim=6; first_obj=top1; folder=""
#optim=SAM; folder_optim=SGD; epochs_optim=6; first_obj=top1; folder="/sgd_with_sam"
#optim=SAM; folder_optim=SAM; epochs_optim=6; first_obj=top1_robust; folder=""
epochs=6
#
#first_obj=robustness
sec_obj=c_params
#
alpha=0.5
pmax=5.0
sigma=0.05
#
seed=1

#export MAGICK_HOME=~/magick
python trainers/entropic/train.py --dataset $dataset \
    --data ../datasets/$dataset --ood_data $ood_data --model mobilenetv3 --device $device \
    --model_path ../results/multires/entropic-mbv3-$dataset-${folder_optim}-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$epochs_optim-multires/final/net-trade-off_0/net.subnet \
    --output_path ../results/multires/entropic-mbv3-$dataset-${folder_optim}-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$epochs_optim-multires/final/net-trade-off_0$folder \
    --pretrained --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --res $res --epochs $epochs --optim $optim --save_ckpt --ood_eval --alpha $alpha \
    --eval_robust

    #--model_path ../results/risultati-res32/entropic-mbv3-$dataset-${folder_optim}-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$epochs_optim-10jan/final/net-trade-off_0/net.subnet \
    #--output_path ../results/risultati-res32/entropic-mbv3-$dataset-${folder_optim}-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$epochs_optim-10jan/final/net-trade-off_0 \