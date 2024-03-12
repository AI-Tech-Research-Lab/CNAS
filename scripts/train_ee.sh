dataset=cifar10
device=0
optim=SGD
backbone_epochs=1

python early_exit/train.py --dataset $dataset  --n_classes 10 \
    --model mobilenetv3 --device $device \
    --model_path ../results/ee_test/net.subnet \
    --output_path ../results/ee_test \
    --pretrained --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --backbone_epochs $backbone_epochs --optim $optim --val_split 0.1 