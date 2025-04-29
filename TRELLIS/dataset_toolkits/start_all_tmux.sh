#!/bin/bash

# 定义会话名称数组"rogland" "sunset" "warm" "zwar" "resting" "blue" "brown" "industrial" "kloppe" "mine"
sessions=( "rogland" "sunset" "warm" "zwar" "resting" )

# 循环遍历会话
for session in "${sessions[@]}"; do
    # 创建一个新的 tmux 会话并在其中执行命令
    echo "creating"
    tmux new-session -d -s "$session" #"bash -i -c 'conda activate trellis; CUDA_VISIBLE_DEVICES=2 python dataset_toolkits/xuxi_making_datasets_all_${session}.py'"

done

echo "所有 tmux 会话已启动。"
