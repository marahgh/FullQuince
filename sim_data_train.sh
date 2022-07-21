#!/bin/bash

quince \
    train \
        --job-dir ~/experiments/quince/ \
        --num-trials 50 \
        --gpu-per-trial 0.2 \
    synthetic \
        --gamma-star 1.65 \
    ensemble \
        --dim-hidden 200 \
        --num-components 5 \
        --depth 4 \
        --negative-slope 0.0 \
        --dropout-rate 0.1 \
        --spectral-norm 6.0 \
        --learning-rate 1e-3 \
        --batch-size 32 \
        --epochs 500 \
        --ensemble-size 10