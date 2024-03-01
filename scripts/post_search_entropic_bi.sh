dataset=cifar10
#dataset=cifar100
#dataset=tinyimagenet
n_epochs=6
#
optim=SGD; first_obj=top1
#optim=SAM; first_obj=top1_robust
#
sec_obj=c_params
pmax=5.0
alpha=0.5
sigma=0.05
#
iter=1

python post_search.py \
  --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
  --get_archive --n 3 --n_classes 10 \
  --save ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$n_epochs-multires-balance/final \
  --expr ../results/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$n_epochs-multires-balance/iter_$iter.stats \
  --first_obj $first_obj --sec_obj $sec_obj #\
  #--sec_obj robustness

  #--save ../results/risultati-res32/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/final \
  #--expr ../results/risultati-res32/entropic-mbv3-$dataset-$optim-$first_obj-$sec_obj-max$pmax-alpha$alpha-sigma$sigma-ep$n_epochs-10jan/iter_$iter.stats \
