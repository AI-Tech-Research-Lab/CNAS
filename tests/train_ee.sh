# Train a model with early exit trainer 

dataset=imagenette
device=0
optim=SGD
backbone_epochs=0

python ee_train.py --dataset $dataset  --n_classes 10 \
    --model eemobilenetv3 --device $device --n_workers 4\
    --model_path results/search_edanas_imagenette/final/net-trade-off_0/net.subnet \
    --output_path results/search_edanas_imagenette/final/net-trade-off_0 \
    --pretrained --supernet_path ./NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --backbone_epochs $backbone_epochs --warmup_ee_epochs 0 --ee_epochs 10 --method joint \
    --optim $optim --val_split 0.1 --save --mmax 100 --top1min 0.1 \
    --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 --resolution 160 --batch_size 128

    