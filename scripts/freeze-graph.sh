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

cd $(dirname $0)/..
ROOT=$(pwd)

TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
TF_TAG=v${TF_VERSION}

FREEZE_GRAPH_URL=https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_TAG}/tensorflow/python/tools/freeze_graph.py
FREEZE_GRAPH_BIN=${ROOT}/scripts/freeze_graph.py

[ ! -f ${FREEZE_GRAPH_BIN} ] && curl -s ${FREEZE_GRAPH_URL} >${FREEZE_GRAPH_BIN}

CHECKPOINT_DIR=$(pwd)/checkpoints

# GRAPH_FILE=${CHECKPOINT_DIR}/graph.pb.txt
# CHECKPOINT=${CHECKPOINT_DIR}/saved_checkpoint-0
# OUTPUT_GRAPH=${CHECKPOINT_DIR}/freezed

OUTPUT_NODE_NAMES=image,upsample_size,upsample_heatmat,tensor_peaks,upsample_pafmat

# usage: freeze <g> <c> <output>, (<g>, <c>) -> <output>
freeze() {
    local GRAPH_FILE=${CHECKPOINT_DIR}/$1
    local CHECKPOINT=${CHECKPOINT_DIR}/$2
    local OUTPUT_GRAPH=${CHECKPOINT_DIR}/$3

    python3 ${FREEZE_GRAPH_BIN} \
        --input_graph ${GRAPH_FILE} \
        --input_checkpoint ${CHECKPOINT} \
        --output_graph ${OUTPUT_GRAPH} \
        --output_node_names ${OUTPUT_NODE_NAMES}
}

export_vgg_graph() {
    measure ./export.py \
        --base-model=vgg \
        --path-to-npz=${HOME}/Downloads/vgg450000.npz \
        --graph-filename='openpose-vgg.pb.txt' \
        --checkpoint-name='openpose-vgg-ckpt'

    measure freeze openpose-vgg.pg.txt openpose-vgg-ckpt-0 openpose-vgg-freezed
}

export_vggtiny_graph() {
    measure ./export.py \
        --base-model=vggtiny \
        --path-to-npz=${HOME}/Downloads/mbn280000.npz \
        --graph-filename='openpose-vggtiny.pb.txt' \
        --checkpoint-name='openpose-vggtiny-ckpt'

    measure freeze openpose-vggtiny.pb.txt openpose-vggtiny-ckpt-0 openpose-vggtiny-freezed
}

# This doesn't work
export_mobilenet_graph() {
    measure ./export.py \
        --base-model=mobilenet \
        --path-to-npz=${HOME}/Downloads/mbn280000.npz \
        --graph-filename='openpose-mobilenet.pb.txt' \
        --checkpoint-name='openpose-mobilenet-ckpt'

    measure freeze openpose-mobilenet.pb.txt openpose-mobilenet-ckpt-0 openpose-mobilenet-freezed
}

[ -d checkpoints ] && rm -fr checkpoints
# export_vgg_graph
export_mobilenet_graph
