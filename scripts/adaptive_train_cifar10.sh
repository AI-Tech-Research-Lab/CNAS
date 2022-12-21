cd ../nsganetv2

python adaptive_train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 20 \
    --cutout --autoaugment \
    --evaluate \
    --model adapt_tinynsganet \
    --model1-config ../results/20221220-155540cifar10adapt_tinynsganet40/net_flops@7.config \
    --model2-config ../results/20221220-160954cifar10adapt_tinynsganet40/net_flops@11.config \
    --initial-checkpoint1 ../results/20221220-155540cifar10adapt_tinynsganet40/net_flops@7.best \
    --initial-checkpoint2 ../results/20221220-160954cifar10adapt_tinynsganet40/net_flops@11.best \
    --drop 0.2 --drop-path 0.2 \
    --img-size 40 --threshold 0.2