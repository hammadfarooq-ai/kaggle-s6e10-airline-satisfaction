#!/bin/bash
# Push the Q-FT kernel only after one of the two ARC3 GPU kernels has finished (do not queue behind them).
export PYTHONUTF8=1
while true; do
  busy=0
  for k in arc3-eval-tc-dprime arc3-bench-throughput; do
    kaggle kernels status malikhammadfarooq/$k 2>&1 | grep -qE 'RUNNING|QUEUED' && busy=$((busy+1))
  done
  if [ $busy -lt 2 ]; then
    echo "$(date -u) ARC3 busy=$busy, pushing"
    kaggle kernels push -p src/qft_transformer --accelerator NvidiaTeslaT4 2>&1 | grep -v Warning
    break
  fi
  sleep 300
done
