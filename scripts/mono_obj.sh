optim=SGD
#optim=SAM
dataset=cifar10

python cnas.py  --resume ../results/entropic-mbv3-$dataset-$optim/iter_0 --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/$dataset --dataset $dataset --predictor as \
              --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --search_space mobilenetv3 \
              --save ../results/entropic-mbv3-$dataset-$optim --iterations 10 --n_epochs 10 \
              --trainer_type entropic \
              --lr 32 --ur 40 --rstep 4 --optim $optim --sigma 0.05
