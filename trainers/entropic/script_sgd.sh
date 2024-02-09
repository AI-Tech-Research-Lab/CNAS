model=mlp
dataset=cifar10
loss=xent
opt=sgd
lrs=(0.001)
seeds=(0)
epochs=200
gpu=0

for lr in ${lrs[@]}; do
    echo $lr
    for seed in ${seeds[@]}; do
        echo $seed
        python ../train.py -F results/${model}_${dataset}_${opt} with \
        project=torchsimple name=${model}_${dataset}_${opt} \
        model=${model} dataset=${dataset} loss=${loss} \
        opt=${opt} lr=${lr} bs=100 wd=0.0 \
        epochs=${epochs} seed=${seed} gpu=${gpu}
    done
done
