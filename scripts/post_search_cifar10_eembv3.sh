#/usr/bin/bash

# last_iter=$(ls ../results/search-wood-params-w1.0/ -v | tail -n1)
# path="../results/search-wood-params-w1.0/$last_iter"
# echo $path

python post_search.py \
  --save ../results/entropic-mbv3-cifar100-SGD-top1-alpha0.9-sigma0.025/final \
  --expr ../results/entropic-mbv3-cifar100-SGD-top1-alpha0.9-sigma0.025/iter_10.stats \
  --n 1 \
  --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
  --search_space cbnmobilenetv3 --lr 32 --ur 32 --rstep 1 \
  --first_obj top1 \
  --n_classes 10 