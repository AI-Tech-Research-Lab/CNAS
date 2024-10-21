# Search a ResNet50 on CIFAR-10 optimizing the average top1 accuracy in presence of drift (single obj)

python cnas.py --resume results/resnet_quant_search/iter_0 \
              --first_obj avg_top1_drift --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 --n_classes 10 \
              --first_predictor as --optim SAM \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_supernet_resnet50 \
              --save results/resnet_quant_search --iterations 30 \
              --search_space resnet50 --trainer_type single_exit \
              --n_epochs 3 --lr 128 --ur 224 --rstep 4 --val_split 0.1 \
              --quantization --drift


            