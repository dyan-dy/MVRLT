#!/bin/bash

# 列出所有tmux session
sessions=$(tmux list-sessions -F "#S")

# 检查有没有session
if [ -z "$sessions" ]; then
    echo "No tmux sessions found."
    exit 0
fi

# 逐个处理
for session in $sessions; do
    echo "Stopping session: $session"

    # 给session发送 Ctrl+C (模拟程序中断)
    tmux send-keys -t "$session" C-c

    # 给程序一点时间来优雅退出
    sleep 2

    # 强制kill掉tmux session
    tmux kill-session -t "$session"
    
    echo "Session $session killed."
done

echo "All tmux sessions have been stopped and killed."
