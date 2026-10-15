#!/bin/bash
# Usage: tools/stage.sh <file>... — copies files into the app's internal files/probe dir
# (adb push to /data/local/tmp, then run-as cp; the app cannot read shell-owned files).
cd "$(dirname "$0")/.."
for f in "$@"; do
  b=$(basename "$f")
  tools/adbr push "$f" /data/local/tmp/$b | tail -1
  tools/adbr shell run-as io.github.kazeevn.kokorotts sh -c "'mkdir -p files/probe && cp /data/local/tmp/$b files/probe/$b'" || exit 1
  tools/adbr shell rm -f /data/local/tmp/$b
done
