#!/bin/bash
# Usage: tools/probe.sh <model-basename-in-probe-dir> <CPU|NPU> [iters] [threads]
# Runs ProbeActivity on the phone and prints its report (survives adb drop-outs).
cd "$(dirname "$0")/.."
M=$1; B=$2; IT=${3:-5}; TH=${4:-4}
D=/sdcard/Android/data/io.github.kazeevn.kokorotts/files/probe
R=/sdcard/Android/data/io.github.kazeevn.kokorotts/files/results/$M.tflite.$B.txt
tools/adbr shell rm -f $R >/dev/null
tools/adbr shell am start -n io.github.kazeevn.kokorotts/.ProbeActivity --es model probe/$M.tflite --es backend $B --ei iters $IT --ei threads $TH >/dev/null
for i in $(seq 1 ${PROBE_TIMEOUT:-600}); do
  if tools/adbr shell "test -f $R && cat $R" 2>/dev/null | grep -q .; then tools/adbr shell cat $R; exit 0; fi
  sleep 1
done
echo "probe timed out"; exit 1
