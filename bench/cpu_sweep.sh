#!/bin/bash
# Timing-only CPU tuning sweep (no power trace): threads x fp16 at the NPU's 128-frame chunking.
cd "$(dirname "$0")/.."
R=/sdcard/Android/data/io.github.kazeevn.kokorotts/files/results
for cfg in "4 false" "4 true" "6 false"; do
  set -- $cfg; tag=sweep_t$1_fp16$2
  tools/cooldown.sh 42 >/dev/null
  tools/adbr shell rm -f $R/bench_$tag.json >/dev/null
  tools/adbr shell am force-stop io.github.kazeevn.kokorotts
  tools/adbr shell am start-foreground-service -n io.github.kazeevn.kokorotts/.BenchService --es backend CPU \
    --ei repeats 1 --ei settle_s 0 --ei frame_cap 128 --ei threads $1 --ez fp16 $2 --es tag $tag >/dev/null
  until tools/adbr shell "test -f $R/bench_$tag.json && echo y" 2>/dev/null | grep -q y; do sleep 5; done
  tools/adbr pull $R/bench_$tag.json bench/results/ >/dev/null
  python3 -c "
import json; d=json.load(open('bench/results/bench_$tag.json'))
p=[x for x in d['phases'] if x['name']=='synth0']
print('$tag', d.get('ok'), d.get('error','') , *(('audio %.1fs wall %.1fs RTF %.3f %s' % (x['audio_s'], x['wall_s'], x['wall_s']/x['audio_s'], x['stage_ms'])) for x in p))"
done
