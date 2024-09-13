#dataset=ImageNet16 val_split=0.2 
dataset=imagenette val_split=0.1
#mode=constraints 
mode=noconstraints
#mode=nopeak

python cnas.py --sec_obj avg_macs --resume results/search_nachos_${mode}_$dataset/iter_1 \
              --n_gpus 1 --gpu 1 --n_workers 4 --seed 42\
              --data datasets/$dataset --dataset $dataset \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/search_nachos_${mode}_$dataset --iterations 9 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --method bernulli --support_set --tune_epsilon \
              --val_split $val_split \
              --n_epochs 5 --warmup_ee_epochs 2 --ee_epochs 3 \
              --w_alpha 1.0 --w_beta 0 --w_gamma 1.0 \
              --lr 160 --ur 160 --rstep 4 

# NACHOS  #[10,5,5] 
# --mmax 2.7 --top1min 0.65 \
