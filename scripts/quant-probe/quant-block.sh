#!/bin/bash
# quant-block.sh <tag> — 健康门禁 + 速度块 + 四室探针
cd /data/ai/quant-probe
for i in $(seq 1 60); do
  curl -s -m 3 http://127.0.0.1:8099/health | grep -q '"status": "ok"' && break
  sleep 5
done
python3 measure.py 8099 "$1" || echo "MEASURE_FAIL_$1"
python3 probe.py 8099 "$1" || echo "PROBE_FAIL_$1"
echo "BLOCK_DONE_$1"
