#!/bin/sh
set -e

cd $(dirname $0)
TF_ROOT=$(pwd)/tensorflow

cd ${TF_ROOT}
cp ../.tf_configure.bazelrc .
echo "import $(pwd)/.tf_configure.bazelrc" >.bazelrc

PKG=tensorflow/examples/pose-inference
bazel build --config=opt --config=cuda //${PKG}:all
