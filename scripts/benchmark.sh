#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")/..";pwd)
export PYTHONPATH=${WORK_DIR}
CHECKPOINT_PATH=$1
DATA_PATH=$2
MODEL_FAMILY=${4:-qwen3_vl}

if [ -z "${CHECKPOINT_PATH}" ] || [ -z "${DATA_PATH}" ]; then
    echo "Error: checkpoint_path and data_path are required"
    echo "Usage: bash scripts/benchmark.sh <checkpoint_path> <data_path> [output_dir] [model_family]"
    exit 1
fi

if [ -n "$3" ]; then
    OUTPUT_DIR_EVAL=$3
    echo "Using custom output directory: ${OUTPUT_DIR_EVAL}"
else
    OUTPUT_DIR_EVAL=$(cd "$(dirname "${CHECKPOINT_PATH}")";pwd)/evaluation
    echo "Using default output directory: ${OUTPUT_DIR_EVAL}"
fi

mkdir -p ${OUTPUT_DIR_EVAL}
CONV_MODE=v1

if [ "${MODEL_FAMILY}" = "qwen3_vl" ]; then
    VQA_LOADER=kve_kd.evaluation.vqa
    MMBENCH_LOADER=kve_kd.evaluation.mmbench
    SCIENCE_LOADER=kve_kd.evaluation.scienceqa
    echo "Using Qwen3-VL evaluation modules"
else
    echo "Error: unsupported model family '${MODEL_FAMILY}'; only qwen3_vl is supported"
    exit 1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 1000 {print $1}' | paste -sd,)
    AVAILABLE_GPUS=$(echo $AVAILABLE_GPUS | sed 's/,$//')
    if [ -n "$AVAILABLE_GPUS" ]; then
        SELECTED_GPUS=${AVAILABLE_GPUS}
        echo "Auto-detected available GPUs: ${SELECTED_GPUS}"
    else
        SELECTED_GPUS="0"
        echo "No idle GPUs detected, using GPU 0 by default"
    fi
else
    SELECTED_GPUS=${CUDA_VISIBLE_DEVICES}
fi

SELECTED_GPUS=$(echo $SELECTED_GPUS | sed 's/,$//' | sed 's/^,//' | sed 's/,,/,/g' | xargs)
if [ -z "$SELECTED_GPUS" ]; then
    SELECTED_GPUS="0"
fi

NUM_GPUS=$(echo $SELECTED_GPUS | tr ',' '\n' | wc -l)
echo "Using ${NUM_GPUS} GPU(s) for benchmark: ${SELECTED_GPUS}"

cd ${WORK_DIR}

DATASET_NAME=mme
MODEL_GENERATOR=${VQA_LOADER}
DATA_ROOT=${DATA_PATH}/benchmark_data/mme
SPLIT_NAME=llava_mme
CUDA_VISIBLE_DEVICES=${SELECTED_GPUS} bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=gqa
MODEL_GENERATOR=${VQA_LOADER}
DATA_ROOT=${DATA_PATH}/benchmark_data/gqa
SPLIT_NAME=llava_gqa_testdev_balanced
CUDA_VISIBLE_DEVICES=${SELECTED_GPUS} bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=textvqa
MODEL_GENERATOR=${VQA_LOADER}
DATA_ROOT=${DATA_PATH}/benchmark_data/textvqa
SPLIT_NAME=llava_textvqa_val_v051_ocr
CUDA_VISIBLE_DEVICES=${SELECTED_GPUS} bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=pope
MODEL_GENERATOR=${VQA_LOADER}
DATA_ROOT=${DATA_PATH}/benchmark_data/pope
SPLIT_NAME=llava_pope_test
CUDA_VISIBLE_DEVICES=${SELECTED_GPUS} bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=mmbench
MODEL_GENERATOR=${MMBENCH_LOADER}
DATA_ROOT=${DATA_PATH}/benchmark_data/mmbench
SPLIT_NAME=mmbench_dev_en_20231003
CUDA_VISIBLE_DEVICES=${SELECTED_GPUS} bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}

DATASET_NAME=sqa
MODEL_GENERATOR=${SCIENCE_LOADER}
DATA_ROOT=${DATA_PATH}/benchmark_data/sqa
SPLIT_NAME=llava_test_CQM-A
CUDA_VISIBLE_DEVICES=${SELECTED_GPUS} bash scripts/benchmark/${DATASET_NAME}.sh \
    ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${DATASET_NAME}
