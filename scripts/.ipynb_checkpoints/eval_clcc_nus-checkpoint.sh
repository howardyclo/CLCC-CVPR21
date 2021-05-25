#!/bin/tcsh

# set BACKBONE="alexnet"
# set MODEL="alexnet_clcc"
set BACKBONE="squeezenet"
set MODEL="squeezenet_clcc"
set DATA_DIR="/proj/gpu_d_03006/howardyclo/CLCC_CVPR20/data"

# =======================================================================
set TEST_DATA="Canon1DsMkIII"

python3 eval.py \
    -data_dir ${DATA_DIR} \
    -test_data ${TEST_DATA} \
    -test_fold 0 \
    -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
    -ckpt_path "../CLCC_CVPR20/backups/NUS0_trainfold12/ckpts/1220.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    -output_dir "./"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
# #     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -ckpt_path "../CLCC_CVPR20/backups/NUS0_trainfold20/ckpts/4220.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
# #     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -ckpt_path "../CLCC_CVPR20/backups/NUS0_trainfold01/ckpts/18160.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# # =======================================================================
# set TEST_DATA="Canon600D"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"

# # =======================================================================
# set TEST_DATA="FujifilmXM1"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# # =======================================================================
# set TEST_DATA="NikonD5200"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# # =======================================================================
# set TEST_DATA="OlympusEPL6"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# # =======================================================================
# set TEST_DATA="PanasonicGX1"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# # =======================================================================
# set TEST_DATA="SamsungNX2000"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# # =======================================================================
# set TEST_DATA="SonyA57"

# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 0 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/12.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 1 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/20.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 eval.py \
#     -data_dir ${DATA_DIR} \
#     -test_data ${TEST_DATA} \
#     -test_fold 2 \
#     -backbone ${BACKBONE} \
#     -ckpt_path "./pretrained_models/nus/${MODEL}/${TEST_DATA}/01.ckpt" \
#     -output_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
# python3 combine.py \
#     -input_dir "inference_results/nus/${MODEL}/${TEST_DATA}"
    
    
