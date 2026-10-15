#!/bin/bash
# Usage: tools/memguard.sh <package> [limit_mb] [seconds]
# Polls the package's RSS on the phone; force-stops it if RSS exceeds the limit (protects the
# user's other apps from the low-memory killer while debugging). Prints the peak RSS seen.
PKG=$1; LIMIT=${2:-1200}; DUR=${3:-120}
peak=0
end=$((SECONDS + DUR))
while [ $SECONDS -lt $end ]; do
  rss=$(adb shell "pid=\$(pidof $PKG); [ -n \"\$pid\" ] && grep VmRSS /proc/\$pid/status" 2>/dev/null | awk '{print int($2/1024)}')
  if [ -n "$rss" ]; then
    [ "$rss" -gt "$peak" ] && peak=$rss
    if [ "$rss" -gt "$LIMIT" ]; then adb shell am force-stop $PKG; echo "KILLED at ${rss} MB"; break; fi
  fi
  sleep 0.3
done
echo "peak RSS ${peak} MB"
