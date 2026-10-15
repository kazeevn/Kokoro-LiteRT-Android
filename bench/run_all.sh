#!/bin/bash
# Full CPU-vs-NPU power benchmark. Each configuration synthesizes bench_corpus.txt (~75 s of
# speech) 3 times after a warm-up pass; idle runs give the baseline. Screen off, foreground service.
set -e
cd "$(dirname "$0")/.."
P=${PREFIX:-r1}
IDLE_S=90 bench/run_bench.sh ${P}_idle_a IDLE
EXTRA="--ei frame_cap 128" bench/run_bench.sh ${P}_cpu128 CPU 3 15
bench/run_bench.sh ${P}_npu NPU 3 15
bench/run_bench.sh ${P}_cpu256 CPU 3 15
IDLE_S=90 bench/run_bench.sh ${P}_idle_b IDLE
.venv-bench/bin/python bench/analyze.py bench/results/${P}_idle_a bench/results/${P}_idle_b \
    bench/results/${P}_cpu128 bench/results/${P}_npu bench/results/${P}_cpu256
