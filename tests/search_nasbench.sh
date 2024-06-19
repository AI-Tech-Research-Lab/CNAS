python cnas.py --n_gpus 1 --gpu 1 --n_workers 4 \
              --data ../datasets/cifar10 --dataset cifar10 \
              --first_predictor gp --optim SGD \
              --pretrained  \
              --save ../results/nasbench-opt_acc --iterations 30 \
              --search_space nasbench --bench_eval --trainer_type single_exit \
              --n_epochs 5 --lr 40 --ur 104 --rstep 4\
              --pmax 2.2 --mmax 7 --amax 0.3 --wp 1.0 --wm 1.0 --wa 1.0 