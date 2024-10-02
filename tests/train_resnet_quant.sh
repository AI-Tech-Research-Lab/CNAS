python train.py --dataset cifar10 --data ../datasets/cifar10 --model resnet50 --device 0 --model_path results/resnet_quant/iter_0/net_0/net_0.subnet \
    --output_path results/resnet_quant/iter_0/net_0 --supernet_path NasSearchSpace/ofa/supernets/ofa_supernet_resnet50 \
    --epochs 1 --batch_size 128 --optim SGD --sigma_min 0.05 --sigma_max 0.05 --sigma_step 0 --alpha 0.5 --res 160 --pmax 2.0 --mmax 10000000000 \
    --amax 5.0 --wp 1.0 --wm 0.025 --wa 1.0 --penalty 10000000000 --alpha_norm 1.0 --val_split 0.1 --n_workers 4 
    

#--quantization --drift