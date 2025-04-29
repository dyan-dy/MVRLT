#!/bin/bash

# 设置内存使用率阈值（百分比）
THRESHOLD=90

# 检查周期（秒）
INTERVAL=10

while true; do
  # 获取 total 和 available 内存（单位 KB）
  read total available <<< $(free | awk '/Mem:/ {print $2, $7}')

  # 计算实际内存使用率 = (总内存 - 可用内存) / 总内存 * 100
  usage=$(( ( (total - available) * 100 ) / total ))

  echo "当前实际内存使用率：${usage}%"

  if [ "$usage" -ge "$THRESHOLD" ]; then
    echo "内存使用率超过 ${THRESHOLD}% ，执行脚本 dataset_toolkits/kill_all_tmux.sh"
    bash dataset_toolkits/kill_all_tmux.sh
    break  # 如果希望持续监控，请移除这一行
  fi

  sleep $INTERVAL
done
