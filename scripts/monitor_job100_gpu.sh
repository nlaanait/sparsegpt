#!/bin/bash
# Sample GPU memory for gjob-100 (Qwen3.5-27B) until it leaves Running.
set -u
LOG=/home/ubuntu/sparsegpt/figures/logs/job100_gpu_monitor.csv
PID_FILE=/tmp/job100_gpu_monitor.pid
echo $$ > "$PID_FILE"
if [[ ! -f "$LOG" ]]; then
  echo "ts_utc,job_state,sparsity_hint,gpu_used_mib,gpu_total_mib,proc_used_mib,gpu_util,host_rss_g" > "$LOG"
fi
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
while true; do
  show=$(gjob show 100 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' || true)
  state=$(printf '%s\n' "$show" | awk -F= '/State=/{print $2; exit}' | awk '{print $1}')
  if [[ "$state" != "Running" ]]; then
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "$ts,$state,,,,,,," >> "$LOG"
    echo "job 100 state=$state — monitor stopping" >> "$LOG"
    break
  fi
  # progress hint from log
  sp=$(gjob log 100 2>/dev/null | grep -E 'sparsity=[0-9.]+ datasets' | tail -1 | sed -n 's/.*sparsity=\([0-9.]*\).*/\1/p')
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  used=$(echo "$gpu" | awk -F', ' '{print $1}')
  total=$(echo "$gpu" | awk -F', ' '{print $2}')
  util=$(echo "$gpu" | awk -F', ' '{print $3}')
  proc=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1==975995{print $2; found=1} END{if(!found) print ""}')
  # refresh python pid if restarted (unlikely)
  if [[ -z "$proc" ]]; then
    proc=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{print $2}')
  fi
  rss=$(ps -o rss= -p 975995 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}')
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$ts,$state,${sp:-},$used,$total,${proc:-},$util,${rss:-}" >> "$LOG"
  sleep 20
done
rm -f "$PID_FILE"
