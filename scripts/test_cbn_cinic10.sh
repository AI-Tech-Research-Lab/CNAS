python cnas.py --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --device 0 --n_workers 4 \
              --dataset cinic10 \
              --predictor as --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save ../results/cinic10-cbn-mbv3-noconstraints --iterations 10 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --lr 32 --ur 32 

#--fmax 2.7 --top1min 0.50 \