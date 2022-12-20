cd ../nsganetv2

python adaptive_train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 20 \
    --cutout --autoaugment --save ../results \
    --evaluate \
    --model adapt_tinynsganet \
    --model1-config ../results/20221220-104528cifar10adapt_tinynsganet40/net_flops@7.config \
    --model2-config ../results/20221220-104528cifar10adapt_tinynsganet40/net_flops@23.config \
    --initial-checkpoint1 ../results/20221220-104528cifar10adapt_tinynsganet40/net_flops@7.best \
    --initial-checkpoint2 ../results/20221220-104528cifar10adapt_tinynsganet40/net_flops@23.best \
    --drop 0.2 --drop-path 0.2 \
    --img-size 40