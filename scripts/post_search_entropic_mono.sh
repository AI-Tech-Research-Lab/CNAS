dataset=cifar10
dataset=cifar100
#dataset=tinyimagenet
n_epochs=10
#
optim=SGD; first_obj=top1
optim=SAM; first_obj=top1_robust
#
alpha=0.9
sigma=0.05
iter=30

python post_search.py \
  --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
  --get_archive --n 1 --n_classes 10 \
  --save ../results/risultati-res32/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/final \
  --expr ../results/risultati-res32/entropic-mbv3-$dataset-$optim-$first_obj-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/iter_$iter.stats \
  --first_obj $first_obj #\
  # --sec_obj $sec_obj