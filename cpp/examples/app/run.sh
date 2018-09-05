#!/bin/sh
set -e

now_nano() {
    date +%s*1000000000+%N | bc
}

measure() {
    echo "[->] $@ begins"
    local begin=$(now_nano)
    $@
    local end=$(now_nano)
    local duration=$(echo "scale=6; ($end - $begin) / 1000000000" | bc)
    echo "[==] $@ took ${duration}s" | tee -a time.log
}

cd $(dirname $0)
SCRIPT_DIR=$(pwd)

cd ${SCRIPT_DIR}/../../..

batch_limit=200
INPUT_IMAGES=$(ls data/media/*.jpg | sort | head -n $batch_limit | tr '\n' ',')
# INPUT_IMAGES=$(ls data/media/*.png | sort | head -n $batch_limit | tr '\n' ',')

measure ${SCRIPT_DIR}/cmake-build/$(uname -s)/bin/see-pose \
    --alsologtostderr \
    --graph_path=checkpoints/openpose-mobilenet-freezed \
    --input_images=${INPUT_IMAGES}
