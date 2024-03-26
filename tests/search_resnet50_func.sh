# Search a ResNet50 on CIFAR-10 optimizing on top1 accuracy and tiny_ml (jointly params, macs, and activations with constraints) with:
# (1) functional constraints: every ReLU activation replaced with OurReLU 
# (2) technological constraints: constraints on params, macs and activations (CNAS)
# OurReLU is a custom activation function implementing a second order polynomial (see utils)

python cnas.py --sec_obj tiny_ml \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 \
              --first_predictor as --optim SGD \
              --supernet_path resnet50 --func_constr \
              --save ../results/resnet_he --iterations 30 \
              --search_space resnet50 --trainer_type single_exit \
              --n_epochs 5 --lr 40 --ur 104 --rstep 4\
              --pmax 0.5 --mmax 150 --amax 5.0 --wp 1.0 --wm 1.0 --wa 1.0