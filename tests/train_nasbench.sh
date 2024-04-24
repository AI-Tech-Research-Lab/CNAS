dataset=cifar10
res=32
device=0
optim=SGD
epochs=200
seed=1

python train.py --dataset $dataset \
    --data ../datasets/$dataset --device $device \
    --output_path ../result/nasbench_train --n_classes 10\
    --res $res --epochs $epochs --optim $optim --alpha $alpha --val_split 0.5 --eval_test \
    --nesterov --weight_decay 0.0005 --momentum 0.9 --learning_rate 0.1 --batch_size 256 