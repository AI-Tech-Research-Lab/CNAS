# Search a MobileNetV3 on CIFAR-10 by optimizing the top1 robust and the number of params with a constraint on the number of params. (FlatNAS)
# Use optimizer SAM in the training of a candidate network

optim=SAM; n_epochs=6; first_obj=top1_robust #optim=SGD
gpu=0
#
dataset=cifar10

sec_obj=tiny_ml
pmax=5.0
wp=1.0
alpha=0.5
sigma=0.05

iterations=30
#
seed=1

lr=128 #min resolution
ur=224 #max resolution
rstep=4 #resolution step

python cnas.py --n_gpus 1 --gpu 1 --gpu_list $gpu --n_workers 4 \
        --data ../datasets/$dataset --dataset $dataset \
        --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --pretrained --search_space mobilenetv3 --trainer_type entropic \
        --alpha $alpha --lr $lr --ur $ur --rstep $rstep --optim $optim \
        --first_obj $first_obj --first_predictor as \
        --sigma_min $sigma --sigma_max $sigma \
        --iterations $iterations --n_epochs $n_epochs --seed $seed \
        --sec_obj $sec_obj --pmax $pmax --wp $wp\
        --save results/search_path 
        