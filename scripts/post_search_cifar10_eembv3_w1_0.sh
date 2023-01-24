#/usr/bin/bash

cd ../nsganetv2
# last_iter=$(ls ../results/search-wood-params-w1.0/ -v | tail -n1)
# path="../results/search-wood-params-w1.0/$last_iter"
# echo $path

python post_search.py \
  --save ../benc/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/final \
  --expr ../results/cifar10-mbv3-adaptive/iter_0.stats \
  -n 1 \
  --supernet_path ./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0 \
  --prefer tiny_ml \
  --save_stats_csv --n_classes 10 \
  --pmax 2.2 --fmax 7 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000
