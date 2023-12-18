#!/usr/bin/env bash

set -x

export ROOT_DIR=`pwd`
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

set +x