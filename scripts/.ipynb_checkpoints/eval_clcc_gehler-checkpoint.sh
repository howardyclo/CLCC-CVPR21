#!/bin/tcsh

# set BACKBONE="alexnet"
# set MODEL="alexnet_clcc"
set BACKBONE="squeezenet"
set MODEL="squeezenet_clcc"

python3 eval.py \
    -data_dir "data" \
    -test_data "gehler" \
    -test_fold 0 \
    -backbone ${BACKBONE} \
    -ckpt_path "./pretrained_models/gehler/${MODEL}/12.ckpt" \
    -output_dir "inference_results/gehler/${MODEL}/"
    
python3 eval.py \
    -data_dir "data" \
    -test_data "gehler" \
    -test_fold 1 \
    -backbone ${BACKBONE} \
    -ckpt_path "./pretrained_models/gehler/${MODEL}/20.ckpt" \
    -output_dir "inference_results/gehler/${MODEL}/"
    
python3 eval.py \
    -data_dir "data" \
    -test_data "gehler" \
    -test_fold 2 \
    -backbone ${BACKBONE} \
    -ckpt_path "./pretrained_models/gehler/${MODEL}/01.ckpt" \
    -output_dir "inference_results/gehler/${MODEL}/"
    
python3 combine.py \
    -input_dir "inference_results/gehler/${MODEL}/"