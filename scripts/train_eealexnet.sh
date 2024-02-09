python main.py +dataset=cifar10 +method=bernulli_logits \
        method.pre_trained=true +model=alexnet +training=cifar10 \
        hydra.run.dir=../results/benchmarks/eealexnet \ 
        training.device=0 experiment.load=true 