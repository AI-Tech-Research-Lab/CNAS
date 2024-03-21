# Train a model with early exit trainer 

dataset=cifar10
device=0
optim=SGD
backbone_epochs=5

python ee_train.py --dataset $dataset  --n_classes 10 \
    --model mobilenetv3 --device $device --threads 4\
    --model_path ../results/nachos-cifar10/iter_0/net_0/net_0.subnet \
    --output_path ../results/nachos-cifar10/iter_0/net_0 \
    --pretrained --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --backbone_epochs $backbone_epochs --warmup_ee_epochs 2 --ee_epochs 3 \
    --optim $optim --val_split 0.1 --learning_rate 0.01 --save --mmax 5 --top1min 70 \
    --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 