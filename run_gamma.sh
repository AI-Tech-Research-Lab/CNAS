cd ../qgamma/src/GAMMA
python main.py --fitness1 EDP --fitness2 EDP --num_pe 16 --l1_size 512 --l2_size 524288 --NocBW 81920000 --offchipBW 81920000 --epochs 300 --model ResNets_model --num_layer 39 --outdir resnet50_ofa --num_pop 50
cd ../../../CNAS