#/usr/bin/bash

cd ../nsganetv2
# last_iter=$(ls ../results/search-wood-params-w1.0/ -v | tail -n1)
# path="../results/search-wood-params-w1.0/$last_iter"
# echo $path

python post_search.py \
  --save ../results/edanas-r32/final \
  --expr ../results/edanas-r32/iter_30.stats \
  -n 1 \
  --supernet_path ./ofa_nets/ofa_eembv3 \
  --prefer trade-off \
  --n_classes 10 \
  --n_exits 4
