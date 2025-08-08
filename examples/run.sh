#!/bin/bash

export PYTHONPATH=../src

RUN_SIM=true
RUN_PLOT=true
for arg in "$@"; do
    if [ "$arg" = "--no-sim" ]; then
        RUN_SIM=false
    elif [ "$arg" = "--no-plot" ]; then
        RUN_PLOT=false
    fi
done

if [ "$RUN_SIM" = true ]; then
    python simulate_data.py --deterministic-sim
fi

python fit_model.py --model-type=full-rank

if [ "$RUN_PLOT" = true ]; then
    python plot_diagnostics.py
fi