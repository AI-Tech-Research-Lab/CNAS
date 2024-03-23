# Search an Early-Exit MobileNetV3 on CIFAR-10 optimizing on top1 accuracy and average number of MACs using a joint trainer on losses (EDANAS)

python cnas.py --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 \
              --first_predictor as --sec_predictor as \
              --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/search_path --iterations 30 \
              --search_space eemobilenetv3 --trainer_type multi_exits \
              --method joint --val_split 0.1 \
              --n_epochs 0  --ee_epochs 5 \
              --lr 40 --ur 104 