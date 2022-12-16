cd ../nsganetv2

python msunas.py  --sec_obj tiny_ml \
              --n_gpus 8 --gpu 1 --n_workers 0 --n_epochs 0 \
              --dataset cifar10 --n_classes 10 \
              --data ../data/cifar10 \
              --predictor as --supernet ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pretrained \
              --save search-cifar10-resnet50 --iterations 1 --vld_size 5000 \
              --pmax 20 --fmax 3000 --amax 33 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000