cd ../nsganetv2

python train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 5 \
    --cutout --autoaugment \
    --save ../results \
    --evaluate \
    --model adapt_tinynsganet \
    --model-config ../results/20221220-154050cifar10adapt_tinynsganet40/net_flops@7.config \
    --initial-checkpoint ../results/20221220-154050cifar10adapt_tinynsganet40/net_flops@7.best \
    --drop 0.2 --drop-path 0.2 \
    --img-size 40