# Look for the n architectures in the archive at iteration iter with the best trade-off between the first and second objectives

# dataset=cifar10 (change num classes accordingly)

first_obj=top1
sec_obj=avg_macs
iter=10
path=results/search_nachos_noconstraints_imagenette

python post_search.py \
  --supernet_path ./NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 \
  --get_archive --n 20 --n_classes 10 \
  --save ${path}/final \
  --expr ${path}/iter_$iter.stats \
  --first_obj $first_obj --sec_obj $sec_obj 