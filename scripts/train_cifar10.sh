cd ../nsganetv2

python train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 150 \
    --cutout --autoaugment --save ../results \
    --model tinynsganet \
    --model-config ../results/search-cifar10-mbv3-w1.0_07_03_2022/final/net-trade-off@1800000011/net.config \
    --initial-checkpoint ../results/search-cifar10-mbv3-w1.0_07_03_2022/final/net-trade-off@1800000011/net.inherited \
    --drop 0.2 --drop-path 0.2 \
    --img-size 192 