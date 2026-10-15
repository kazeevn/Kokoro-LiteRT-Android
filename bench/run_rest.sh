#!/bin/bash
# Remaining configurations of run_all.sh (after an interrupted session).
cd "$(dirname "$0")/.."
P=${PREFIX:-r1}
bench/run_bench.sh ${P}_npu NPU 3 15
bench/run_bench.sh ${P}_cpu256 CPU 3 15
IDLE_S=90 bench/run_bench.sh ${P}_idle_b IDLE
