
python evaluator.py --subnet ./net_small.subnet \
--data ../data/cifar10 --dataset cifar10 \
--n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained \
 --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 \
 --resolution 224 --valid_size 5000 --reset_running_statistics