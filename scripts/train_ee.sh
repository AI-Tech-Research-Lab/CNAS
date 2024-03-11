dataset=cifar10
device=0
optim=SGD
epochs=5

python early_exit/train.py --dataset $dataset \
    --data ../datasets/$dataset --model mobilenetv3 --device $device \
    --model_path ../results/ee_test/net.subnet \
    --output_path ../results/ee_test \
    --pretrained --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --epochs $epochs --optim $optim --val_fraction 0.1 \
    --fix_last_layer