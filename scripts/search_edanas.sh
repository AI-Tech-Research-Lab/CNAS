# Search an Early-Exit MobileNetV3 on CIFAR-10 optimizing on top1 accuracy and average number of MACs using a joint trainer on losses (EDANAS)

dataset=cifar10 val_split=0.1 img_size=32
#dataset=imagenette val_split=0.1 img_size=160
seed=1


python cnas.py --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4 --seed $seed \
              --data datasets/$dataset --dataset $dataset \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/search_edanas_dataset${dataset}_seed$seed --iterations 30 \
              --search_space eemobilenetv3 --trainer_type multi_exits \
              --method joint --joint_type losses --val_split $val_split \
              --n_epochs 0  --ee_epochs 5 \
              --lr $img_size --ur $img_size --rstep 4
