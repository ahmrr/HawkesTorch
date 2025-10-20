#!/bin/bash

SIM_ARGS="--deterministic-sim --M 500 --N 65536 --T 10000"
FIT_ARGS="--model-type=full-rank --M 500 --N 65536 --T 10000"

set -euo pipefail

RUN_SIM=true
RUN_FIT=true
RUN_PLOT=true
SAVE_LOG=true

for arg in "$@"; do
    case "$arg" in
        --no-sim)   RUN_SIM=false ;;
        --no-fit)   RUN_FIT=false ;;
        --no-plot)  RUN_PLOT=false ;;
        --no-log)   SAVE_LOG=false ;;
    esac
done

if $SAVE_LOG; then
    exec > >(tee run.log) 2>&1
fi

export PYTHONPATH=../src

declare -A commands=(
    [sim]="python simulate_data.py $SIM_ARGS"
    [fit]="python fit_model.py $FIT_ARGS"
    [plot]="python plot_diagnostics.py"
)

$RUN_SIM  && ${commands[sim]}
$RUN_FIT  && ${commands[fit]}
$RUN_PLOT && ${commands[plot]}
