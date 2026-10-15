#!/bin/bash
# Power benchmark: one Perfetto power-rail trace per BenchService run.
# Usage: bench/run_bench.sh <tag> <CPU|NPU|IDLE> [repeats] [settle_s]
#   EXTRA="--ei frame_cap 128 --ez fp16 true" passes extra BenchService extras.
# Results: bench/results/<tag>.json (phase timestamps) + <tag>.pftrace
set -e
cd "$(dirname "$0")/.."
TAG=$1; BACKEND=$2; REPEATS=${3:-3}; SETTLE=${4:-15}
A=tools/adbr; PKG=io.github.kazeevn.kokorotts
OUT=bench/results; mkdir -p $OUT
RES=/sdcard/Android/data/$PKG/files/results/bench_$TAG.json
TRACE=/data/misc/perfetto-traces/kokoro_$TAG.pftrace

tools/cooldown.sh ${COOL_C:-42}
$A shell rm -f $RES $TRACE >/dev/null
$A push bench/power_rails.cfg /data/local/tmp/power_rails.cfg >/dev/null
$A shell "cat /data/local/tmp/power_rails.cfg | perfetto --background --txt -c - -o $TRACE" | tail -1
sleep 3
$A shell input keyevent KEYCODE_SLEEP   # screen off, like listening with the phone locked
if [ "$BACKEND" = IDLE ]; then
  $A shell am start-foreground-service -n $PKG/.BenchService --es mode idle --ei idle_s ${IDLE_S:-90} --es tag $TAG >/dev/null
else
  $A shell am start-foreground-service -n $PKG/.BenchService --es backend $BACKEND --ei repeats $REPEATS --ei settle_s $SETTLE --es tag $TAG $EXTRA >/dev/null
fi
for i in $(seq 1 ${BENCH_TIMEOUT:-3600}); do
  if $A shell "test -f $RES && echo yes" 2>/dev/null | grep -q yes; then break; fi
  sleep 2
done
sleep 3
$A shell "pkill -INT -x perfetto" || true
sleep 5
$A pull $RES $OUT/$TAG.json >/dev/null
$A pull $TRACE $OUT/$TAG.pftrace >/dev/null
$A shell input keyevent KEYCODE_WAKEUP
echo "done: $OUT/$TAG.json $OUT/$TAG.pftrace"
