# Train a model with robustness trainer and evaluate on the clean data and eventually on the ood data

dataset=cifar10; res=180; ood_data="../datasets/cifar10c"

device=0

optim=SAM; folder_optim=SAM; epochs_optim=6; first_obj=top1_robust; folder=""
epochs=5

sec_obj=c_params
#
alpha=0.5
pmax=5.0
sigma=0.05
#
seed=1

python robustness/train.py --dataset $dataset \
    --data datasets/$dataset --ood_data $ood_data --model mobilenetv3 --device $device \
    --model_path results/entropic-mbv3-$dataset-${folder_optim}-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$epochs_optim-multires/final/net-trade-off_0/net.subnet \
    --output_path results/entropic-mbv3-$dataset-${folder_optim}-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$epochs_optim-multires/final/net-trade-off_0$folder \
    --pretrained --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --n_classes 10\
    --res $res --epochs $epochs --optim $optim --alpha $alpha --use_val --eval_test