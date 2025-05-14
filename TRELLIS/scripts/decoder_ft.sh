#!/bin/bash

# 设置环境变量（根据实际需要）
export CUDA_VISIBLE_DEVICES=0 #,1,2  # 指定使用哪些GPU
# export OMP_NUM_THREADS=8         # 多线程优化（可选）

# 配置路径
CONFIG=configs/vae/relit_slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json           # JSON配置文件路径
OUTPUT_DIR=outputs/ft        # 输出目录
DATA_DIR=datasets/Proc_Data                            # 数据集目录
# LOAD_DIR=                                      # 可选加载目录（为空时默认为OUTPUT_DIR）

# 启动训练
python train_relit.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    # --load_dir $LOAD_DIR \
    # --ckpt latest \
    --num_gpus 1 \
    --num_nodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 12345 \
    --auto_retry 3 \
    > $OUTPUT_DIR/train.log 2>&1
