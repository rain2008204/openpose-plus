#!/bin/sh
set -e
cd $(dirname $0)/..

export PYTHONUNBUFFERED=1

profile_model() {
    ./test_inference.py --path-to-npz=$HOME/Downloads/$2 \
        --base-model=$1 \
        --images=$(ls data/media/*.jpg | sort | tr '\n' ',') \
        --repeat 1 \
        --plot=False \
        >logs/$1.stdout.log 2>logs/$1.stderr.log
}

mkdir -p logs
profile_model vgg vgg450000_no_cpm.npz
profile_model vggtiny pose195000.npz
profile_model mobilenet mbn280000.npz
