cd ../nsganetv2

python train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 20 \
    --cutout --autoaugment \
    --save ../results \
    --model adapt_tinynsganet \
    --model-config ../benchmarks/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/final/net-trade-off@22740000005/net.config \
    --drop 0.2 --drop-path 0.2 \
    --img-size 40