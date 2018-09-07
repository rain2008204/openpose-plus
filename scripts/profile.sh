#!/bin/sh
set -e
cd $(dirname $0)/..

profile_model() {
    ./test_inference.py --path-to-npz=$HOME/Downloads/$2 \
        --base-model=$1 \
        --images=$(ls data/media/*.jpg | sort | tr '\n' ',') \
        --plot=False
}

profile_model vgg vgg450000_no_cpm.npz >vgg.log
profile_model vggtiny pose195000.npz >tinyvgg.log
profile_model mobilenet mbn28000.npz >mobilenet.log
