#!/bin/bash

for ((i=1; i<=20; i++)); do
    echo "第 $i 次执行"
    python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
    python dataset_toolkits/download.py HSSD --output_dir datasets/HSSD --rank 0 --world_size 2
    sleep 1  # 可选
done