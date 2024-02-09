model=ofa_mbv3
dataset=cifar10
loss=xent
opt=sam
rho=0.05
adaptive=False
lr=0.001
seed=0
epochs=200
gpu=0

python train_entropic.py -F results/${model}_${dataset}_${opt} with \
project=torchsimple name=${model}_${dataset}_${opt} \
model=${model} dataset=${dataset} loss=${loss} \
opt=${opt} lr=${lr} bs=100 wd=0.0 rho=${rho} adaptive=${adaptive} \
epochs=${epochs} seed=${seed} gpu=${gpu}

