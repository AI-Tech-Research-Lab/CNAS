#dataset=ImageNet16 val_split=0.2 
dataset=imagenette val_split=0.1

python cnas.py --sec_obj avg_macs --resume results/search_edanas_$dataset/iter_0 \
              --n_gpus 1 --gpu 1 --n_workers 4 --seed 42 \
              --data datasets/$dataset --dataset $dataset \
              --first_predictor mlp --sec_predictor mlp \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/search_edanas_$dataset --iterations 30 \
              --search_space eemobilenetv3 --trainer_type multi_exits \
              --method joint --val_split $val_split \
              --n_epochs 0  --ee_epochs 5 \
              --lr 160 --ur 160 --rstep 4