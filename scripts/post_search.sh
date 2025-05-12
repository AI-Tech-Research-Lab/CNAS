# Look for the n architectures in the archive at iteration iter with the best trade-off between the first and second objectives

# dataset=cifar10 (change num classes accordingly)

first_obj=top1
sec_obj=avg_macs
iter=10
folder=results/nachos_cifar10_noconstraints_seed1

python post_search.py \
  --get_archive --n 10 \
  --save $folder/final \
  --expr $folder/iter_$iter.stats \
  --first_obj $first_obj --sec_obj $sec_obj 