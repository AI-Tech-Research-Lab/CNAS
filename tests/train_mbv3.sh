epochs=200
python train.py --dataset cifar10 --data datasets/cifar10 --model mobilenetv3 --device 0 \
    --model_path results/search_benchmark_macs/final/net-top1_0/net.subnet \
    --output_path results/search_benchmark_macs/final/net-top1_0 --pretrained \
    --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --epochs $epochs \
    --batch_size 256 --optim SGD --sigma_min 0.05 --sigma_max 0.05 --sigma_step 0 \
    --alpha 0.5 --res 32 --pmax 2.2 --mmax 7.3 --amax 0.3 --wp 0.0 --wm 1.0 --wa 0.0 --penalty 10000000000 \
    --alpha_norm 1.0 --val_split 0.0 --n_workers 4 --n_classes 1000