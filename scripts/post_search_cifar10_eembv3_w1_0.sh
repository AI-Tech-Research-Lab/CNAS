#/usr/bin/bash

cd ../nsganetv2
# last_iter=$(ls ../results/search-wood-params-w1.0/ -v | tail -n1)
# path="../results/search-wood-params-w1.0/$last_iter"
# echo $path

python post_search.py \
  --save ../results/cifar10-mbv3-adaptive/final \
  --expr ../results/cifar10-mbv3-adaptive/iter_30.stats \
  -n 10 \
  --supernet_path ./ofa_nets/ofa_eembv3 \
  --prefer macs \
  --save_stats_csv \
  --n_classes 10 \
  --n_exits 4
