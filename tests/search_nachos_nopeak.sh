#dataset=ImageNet16 val_split=0.2 
dataset=imagenette val_split=0.1
#mode=constraints 
#mode=noconstraints
mode=nopeak

python cnas.py --sec_obj avg_macs \
              --n_gpus 1 --gpu 1 --n_workers 4 --seed 42\
              --data datasets/$dataset --dataset $dataset \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/search_nachos_${mode}_$dataset --iterations 10 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --method bernulli --tune_epsilon \
              --val_split $val_split \
              --n_epochs 5 --warmup_ee_epochs 0 --ee_epochs 5 \
              --w_alpha 1.0 --w_beta 1.0 --w_gamma 0 \
              --lr 160 --ur 160 --rstep 4 --mmax 50 --top1min 0.70