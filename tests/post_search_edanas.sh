# Look for the n architectures in the archive at iteration iter with the best trade-off between the first and second objectives

# dataset=cifar10 (change num classes accordingly)

first_obj=top1
sec_obj=avg_macs
iter=30

python post_search.py \
  --supernet_path ./NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 \
  --get_archive --n 1 --n_classes 10 \
  --save results/search_edanas_imagenette/final \
  --expr results/search_edanas_imagenette/iter_$iter.stats \
  --first_obj $first_obj --sec_obj $sec_obj 