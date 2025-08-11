#!/bin/bash

set -e

export PYTHONPATH=../src

RUN_SIM=true
RUN_PLOT=true
SAVE_LOG=true
for arg in "$@"; do
    case "$arg" in
        --no-sim)   RUN_SIM=false ;;
        --no-plot)  RUN_PLOT=false ;;
        --no-log)   SAVE_LOG=false ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

exec > >(tee run.log) 2>&1

if [ "$RUN_SIM" = true ]; then
    python simulate_data.py --deterministic-sim
fi

python fit_model.py --model-type=full-rank

if [ "$RUN_PLOT" = true ]; then
    python plot_diagnostics.py
fi