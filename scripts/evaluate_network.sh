
python evaluator.py --subnet ../benchmarks/tiny_ml/search-cifar10-mbv3-w1.0-2022-03-23/iter_0/net_3.subnet \
--data ../data/cifar10 --dataset cifar10 \
--n_classes 10 --supernet ./ofa_nets/ofa_eembv3_d234_e346_k357_w1.0 --pretrained \
--trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 \
--resolution 224 --valid_size 5000 --reset_running_statistics