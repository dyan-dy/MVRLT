#!/bin/bash

# 要监听的 tmux 会话名
session_name="scsp"

# 检查 tmux 会话是否存在
while tmux has-session -t $session_name 2>/dev/null; do
    # 捕获该会话的最后 100 行输出
    output=$(tmux capture-pane -t $session_name -pS -100)

    # 检查是否有 "scp -r" 命令正在执行
    if echo "$output" | grep -q -E "\.png|\.json";then
        # 如果 scp 进程正在运行，等待 10 秒后继续检查
        echo "still running"
        sleep 30
    else
        # 如果 scp 进程已经结束，执行 start_all_tmux.py
        echo "scp 传输程序执行完毕，启动 start_all_tmux.py"
        mv -r image_datasets/HSSD/ image_datasets/already_done
        bash dataset_toolkits/start_all_tmux.sh
        break
    fi
done
