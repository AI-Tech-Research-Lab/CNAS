# Train a model with early exit trainer 

dataset=imagenette
device=0
optim=SGD
backbone_epochs=
ofa=False

python ee_train.py --dataset $dataset  --n_classes 10 \
    --model efficientnet --device $device --n_workers 4\
    --output_path results/efficientnet \
    --optim $optim --val_split 0.1  \
    --resolution 160 --batch_size 128  \
    --backbone_epochs 5 --warmup_ee_epochs 0 --ee_epochs 5  --method bernulli --tune_epsilon \
    --w_alpha 1.0 --w_beta 0 --w_gamma 0 