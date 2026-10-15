#!/bin/bash
# Wait until the phone's hottest CPU cluster is below ${1:-45} C (thermal throttling skews benchmarks).
cd "$(dirname "$0")/.."
LIMIT=${1:-45}
for i in $(seq 1 ${COOLDOWN_TIMEOUT:-900}); do
  t=$(tools/adbr shell dumpsys thermalservice 2>/dev/null | sed -n '/Current temperatures from HAL/,/Current cooling/p' \
      | grep -E "mName=(BIG|MID|LITTLE|TPU)," | sed -E 's/.*mValue=([0-9.]+).*/\1/' | sort -n | tail -1)
  [ -n "$t" ] && awk "BEGIN{exit !($t < $LIMIT)}" && { echo "cool: ${t}C"; exit 0; }
  sleep 5
done
echo "cooldown timeout (last ${t}C)"; exit 1
