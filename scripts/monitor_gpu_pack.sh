#!/bin/bash
# Sample GPU packing while shared likelihood jobs run.
set -u
LOG=/home/ubuntu/sparsegpt/figures/logs/gpu_pack_monitor.csv
echo $$ > /tmp/gpu_pack_monitor.pid
[[ -f "$LOG" ]] || echo "ts_utc,n_compute_apps,gpu_used_mib,gpu_free_mib,gpu_util,app_mems_mib,running_jobs" > "$LOG"
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
while true; do
  q=$(gqueue 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')
  nrun=$(printf '%s\n' "$q" | awk '$3=="R" || $0 ~ / R /{c++} END{print c+0}')
  runs=$(printf '%s\n' "$q" | awk '$0 ~ / R /{printf "%s ", $1}')
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1)
  used=$(echo "$gpu" | awk -F', ' '{print $1}')
  free=$(echo "$gpu" | awk -F', ' '{print $2}')
  util=$(echo "$gpu" | awk -F', ' '{print $3}')
  apps=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)
  napp=$(printf '%s\n' "$apps" | awk 'NF{c++} END{print c+0}')
  mems=$(printf '%s\n' "$apps" | awk -F', ' 'NF{printf "%s:%s ", $1,$2}')
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$ts,$napp,$used,$free,$util,${mems% },${runs% }" >> "$LOG"
  # stop when queue empty of Running/Queued (only Hold left or empty)
  nactive=$(printf '%s\n' "$q" | awk 'NR>1 && $3!="H" && NF{c++} END{print c+0}')
  [[ "$nactive" -eq 0 ]] && break
  sleep 20
done
rm -f /tmp/gpu_pack_monitor.pid
