# Search a ResNet50 on CIFAR-10 optimizing on top1 accuracy and tiny_ml (jointly params, macs, and activations with constraints) with:
# (1) functional constraints: every ReLU activation replaced with OurReLU 
# (2) technological constraints: constraints on params, macs and activations (CNAS)
# OurReLU is a custom activation function implementing a second order polynomial (see utils)

python cnas.py --sec_obj params \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 \
              --first_predictor as --optim SGD \
              --supernet_path resnet50  --quantization \
              --save results/resnet_quant --iterations 30 \
              --search_space resnet50 --trainer_type single_exit \
              --n_epochs 5 --lr 32 --ur 32 --rstep 4 --val_split 0.1 