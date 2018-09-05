#!/bin/sh
set -e

# make
# ./cpp/examples/app/run.sh

batch_limit=200
INPUT_IMAGES=$(ls data/media/*.jpg | sort | head -n $batch_limit | tr '\n' ',')

./test_inference.py \
    --path-to-npz=$HOME/Downloads/vggtiny_hao18_pose195000.npz \
    --images=${INPUT_IMAGES} \
    --base-model=vggtiny

# ./test_inference-2.py \
#     --path-to-freezed-model=checkpoints/openpose-mobilenet-freezed \
#     --images=${INPUT_IMAGES}
