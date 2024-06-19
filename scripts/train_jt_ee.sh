# Train a model with early exit joint trainer

dataset=cifar10
device=0
optim=SGD
backbone_epochs=5

python early_exit/train.py --dataset $dataset --data datasets/$dataset --n_classes 10 \
    --model mobilenetv3 --device $device --threads 4\
    --model_path results/ee_test/net.subnet \
    --output_path results/ee_test \
    --pretrained --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --backbone_epochs $backbone_epochs --ee_epochs 3 \
    --method joint --optim $optim --use_val --save 