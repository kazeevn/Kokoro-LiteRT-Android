#!/bin/bash
# Usage: tools/nnprobe.sh <path/to/model.tflite> [accel=google-edgetpu|cpu] [iters]
# Runs the throwaway NNAPI probe app on a model; prints result + darwinn HAL diagnostics.
cd "$(dirname "$0")/.."
F=$1; ACC=${2:-google-edgetpu}; IT=${3:-5}; B=$(basename "$F")
P=io.github.kazeevn.nnprobe
R=/sdcard/Android/data/$P/files/results/$B.nnapi-$ACC.txt
tools/adbr push "$F" /data/local/tmp/$B >/dev/null
tools/adbr shell run-as $P sh -c "'mkdir -p files/probe && cp /data/local/tmp/$B files/probe/$B'" >/dev/null
tools/adbr shell rm -f $R /data/local/tmp/$B >/dev/null
tools/adbr logcat -c
tools/adbr shell am start -n $P/.NnProbeActivity --es model $B --es accel $ACC --ei iters $IT --ez cache ${CACHE:-false} --ez fp16 ${FP16:-true} ${SIG:+--es sig $SIG} >/dev/null
for i in $(seq 1 ${PROBE_TIMEOUT:-300}); do
  out=$(tools/adbr shell "test -f $R && cat $R" 2>/dev/null)
  [ -n "$out" ] && break; sleep 1
done
echo "$out"
tools/adbr logcat -d | grep -E "darwinn-aidl" | grep -E "RET_CHECK|not supported|Ops supported|TPU Subgraphs|delegated to CPU|rror" \
  | sed -E 's/^.*darwinn-aidl: //; s#vendor/google_devices/[^ ]*/third_party/##' | sort | uniq -c | sort -rn | head -${NLOG:-8}
