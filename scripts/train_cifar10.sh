cd ../nsganetv2

python adaptive_train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 20 \
    --cutout --autoaugment --save ../results \
    --evaluate \
    --model adapt_tinynsganet \
    --model1-config ../benchmarks/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/final/net-tiny_ml@3/net.config \
    --model2-config ../benchmarks/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/final/net-tiny_ml@93010000012/net.config \
    --initial-checkpoint1 ../benchmarks/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/final/net-tiny_ml@3/net.inherited \
    --initial-checkpoint2 ../benchmarks/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/final/net-tiny_ml@93010000012/net.inherited \
    --drop 0.2 --drop-path 0.2 \
    --img-size 40