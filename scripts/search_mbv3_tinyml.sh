# Search a MobileNetV3 on CIFAR-10 optimizing on top1 accuracy and tiny_ml (jointly params, macs, and activations with constraints). (CNAS)

python cnas.py --sec_obj tiny_ml \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 \
              --first_predictor as --optim SGD \
              --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/search_path --iterations 30 \
              --search_space mobilenetv3 --trainer_type single_exit \
              --n_epochs 5 --lr 40 --ur 104 --rstep 4\
              --pmax 2.2 --mmax 7 --amax 0.3 --wp 1.0 --wm 1.0 --wa 1.0 