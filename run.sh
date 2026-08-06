#!/bin/bash

WORK_DIR=$(cd "$(dirname "$0")";pwd)
export PYTHONPATH=${WORK_DIR}
export CUDA_HOME=${CONDA_PREFIX}
export PYTHONWARNINGS='ignore::UserWarning,ignore::FutureWarning'
export TOKENIZERS_PARALLELISM=false

python -c "import torch; torch.cuda.init()" 2>/dev/null || true

# Model Path Configuration
STUDENT_MODEL=${STUDENT_MODEL:-}
TEACHER_MODEL=${TEACHER_MODEL:-}

# Data Path Configuration
DATA_PATH=${DATA_PATH:-}
PRETRAIN_DATA_JSON=${PRETRAIN_DATA_JSON:-pretrain.json}
FINETUNE_DATA_JSON=${FINETUNE_DATA_JSON:-finetune.json}

# Output Directory Configuration
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-${WORK_DIR}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# GPU Configuration
SELECTED_GPUS=${CUDA_VISIBLE_DEVICES:-0}
SELECTED_GPUS=$(echo $SELECTED_GPUS | sed 's/,$//' | sed 's/^,//' | sed 's/,,/,/g' | xargs)
if [ -z "$SELECTED_GPUS" ]; then
    SELECTED_GPUS="0"
fi

NUM_GPUS=$(echo $SELECTED_GPUS | tr ',' '\n' | wc -l)

DS_INCLUDE="localhost:${SELECTED_GPUS}"

export CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}

# Batch Configuration
PRETRAIN_BS=${PRETRAIN_BS:-4}
FINETUNE_BS=${FINETUNE_BS:-4}
DEFAULT_PRETRAIN_GAS=$((64 / (NUM_GPUS * PRETRAIN_BS)))
DEFAULT_FINETUNE_GAS=$((128 / (NUM_GPUS * FINETUNE_BS)))
[ ${DEFAULT_PRETRAIN_GAS} -lt 1 ] && DEFAULT_PRETRAIN_GAS=1
[ ${DEFAULT_FINETUNE_GAS} -lt 1 ] && DEFAULT_FINETUNE_GAS=1
PRETRAIN_GAS=${PRETRAIN_GAS:-${DEFAULT_PRETRAIN_GAS}}
FINETUNE_GAS=${FINETUNE_GAS:-${DEFAULT_FINETUNE_GAS}}

# Distillation Parameter Configuration
DISTILL=${DISTILL:-1}  # Enable distillation: 1=enabled, 0=disabled
ADAPTIVE_TOPK=${ADAPTIVE_TOPK:-true}
TOPK_MIN=${TOPK_MIN:-16}
TOPK_MAX=${TOPK_MAX:-64}
TOPK_GAMMA=${TOPK_GAMMA:-1.0}
TOPK_FIXED=${TOPK_FIXED:-16}
TOPK_ATTN_LAYER_IDX=${TOPK_ATTN_LAYER_IDX:-0}  # Attention layer index for Top-K selection: 0=first, -1=last
USE_ENTROPY_WEIGHTING=${USE_ENTROPY_WEIGHTING:-false}

LAMBDA_RKLD=${LAMBDA_RKLD:-0.05}
LAMBDA_V_ALL=${LAMBDA_V_ALL:-0.0}
LAMBDA_V_FOCUS=${LAMBDA_V_FOCUS:-0.5}
LAMBDA_ATTN=${LAMBDA_ATTN:-0.0}

# Key Visual Token Selection Configuration
KEY_VIS_TOKEN_METHOD=${KEY_VIS_TOKEN_METHOD:-kve_topk}

KEY_VIS_P=${KEY_VIS_P:-0.01}  
KEY_VIS_LAYER_THRESHOLD=${KEY_VIS_LAYER_THRESHOLD:-0.99}  
KEY_VIS_TOKENS_THRESHOLD=${KEY_VIS_TOKENS_THRESHOLD:-0.2}  
KEY_VIS_DEBUG=${KEY_VIS_DEBUG:-false} 

KEY_VIS_TRIGGER_METRIC=${KEY_VIS_TRIGGER_METRIC:-cosine}              
KEY_VIS_REL_L2_LAYER_THRESHOLD=${KEY_VIS_REL_L2_LAYER_THRESHOLD:-0.05} 
KEY_VIS_TRIGGER_SCAN_START_LAYER=${KEY_VIS_TRIGGER_SCAN_START_LAYER:-0} 
KEY_VIS_KVE_QUERY_MODE=${KEY_VIS_KVE_QUERY_MODE:-last_text_token}      

# Mixed Attention Configuration (enabled by default)
ENABLE_MIXED_ATTN=${ENABLE_MIXED_ATTN:-true}

# LoRA Configuration (optional, disabled by default)
LORA_ENABLE=${LORA_ENABLE:-false}
LORA_ENABLE_PT=${LORA_ENABLE_PT:-${LORA_ENABLE}}
LORA_ENABLE_FT=${LORA_ENABLE_FT:-${LORA_ENABLE}}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-none}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"}

# Training Configuration
DEEPSPEED_CONFIG_PT=${DEEPSPEED_CONFIG_PT:-scripts/deepspeed/zero2.json}
DEEPSPEED_CONFIG_FT=${DEEPSPEED_CONFIG_FT:-scripts/deepspeed/zero2.json}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-false}  
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
SAVE_TOTAL_LIMIT_FT=${SAVE_TOTAL_LIMIT_FT:-5}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
PRETRAIN_LR=${PRETRAIN_LR:-2e-5}
FINETUNE_LR=${FINETUNE_LR:-4e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-1e-6}  
MM_PROJECTOR_LR_PT=${MM_PROJECTOR_LR_PT:-${MM_PROJECTOR_LR}}  
MM_PROJECTOR_LR_FT=${MM_PROJECTOR_LR_FT:-${MM_PROJECTOR_LR}}  
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}  
RANDOM_SEED=${RANDOM_SEED:-42}  

# Image Pixel Configuration
MIN_PIXELS=${MIN_PIXELS:-$((256*28*28))}
MAX_PIXELS=${MAX_PIXELS:-$((788*28*28))}

# DataLoader / Sampler Configuration
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
GROUP_BY_MODALITY_LENGTH=${GROUP_BY_MODALITY_LENGTH:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
DATALOADER_DROP_LAST=${DATALOADER_DROP_LAST:-True}  

# Task Selection
TASK=${1:-pretrain}

validate_training_inputs() {
    local missing=()
    [ -n "${STUDENT_MODEL}" ] || missing+=("STUDENT_MODEL")
    [ -n "${DATA_PATH}" ] || missing+=("DATA_PATH")
    if [ "${DISTILL}" = "1" ] && [ -z "${TEACHER_MODEL}" ]; then
        missing+=("TEACHER_MODEL")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Error: required environment variable(s) not set: ${missing[*]}" >&2
        echo "Set model and data paths before running KVE-KD training." >&2
        exit 1
    fi
}

case "$TASK" in
    pretrain-finetune)
        validate_training_inputs
        OUTPUT_DIR=${OUTPUT_BASE_DIR}/outputs/kve_kd_qwen3_vl_${TIMESTAMP}
        mkdir -p ${OUTPUT_DIR}
        
        LORA_ENABLE=${LORA_ENABLE_PT} bash $0 pretrain ${OUTPUT_DIR}
        
        echo ">>> Waiting for GPU memory release before finetune..."
        sleep 30
        
        if [ "${LORA_ENABLE_PT}" = "true" ]; then
            echo ">>> Merging pretrain LoRA weights..."
            PRETRAIN_OUTPUT_DIR=${OUTPUT_DIR}/pretrain
            MERGED_OUTPUT_DIR=${OUTPUT_DIR}/pretrain_merged
            python ${WORK_DIR}/scripts/merge_lora_qwen3_vl.py \
                ${STUDENT_MODEL} \
                ${PRETRAIN_OUTPUT_DIR} \
                ${MERGED_OUTPUT_DIR}
            export STUDENT_MODEL=${MERGED_OUTPUT_DIR}
            export STUDENT_MODEL_FROM_PIPELINE=1
        fi
        
        LORA_ENABLE=${LORA_ENABLE_FT} bash $0 finetune ${OUTPUT_DIR}
        
        echo "Pipeline completed. Output saved to: ${OUTPUT_DIR}"
        ;;
    
    pretrain-finetune-test)
        validate_training_inputs
        OUTPUT_DIR=${OUTPUT_BASE_DIR}/outputs/kve_kd_qwen3_vl_${TIMESTAMP}
        mkdir -p ${OUTPUT_DIR}
        
        LORA_ENABLE=${LORA_ENABLE_PT} bash $0 pretrain ${OUTPUT_DIR}
        
        echo ">>> Waiting for GPU memory release before finetune..."
        sleep 30
        
        if [ "${LORA_ENABLE_PT}" = "true" ]; then
            echo ">>> Merging pretrain LoRA weights..."
            PRETRAIN_OUTPUT_DIR=${OUTPUT_DIR}/pretrain
            MERGED_OUTPUT_DIR=${OUTPUT_DIR}/pretrain_merged
            python ${WORK_DIR}/scripts/merge_lora_qwen3_vl.py \
                ${STUDENT_MODEL} \
                ${PRETRAIN_OUTPUT_DIR} \
                ${MERGED_OUTPUT_DIR}
            export STUDENT_MODEL=${MERGED_OUTPUT_DIR}
            export STUDENT_MODEL_FROM_PIPELINE=1
        fi
        
        LORA_ENABLE=${LORA_ENABLE_FT} bash $0 finetune ${OUTPUT_DIR}
        
        echo ">>> Waiting for GPU memory release before test..."
        sleep 30
        
        FINETUNE_OUTPUT_DIR=${OUTPUT_DIR}/finetune
        if [ "${LORA_ENABLE_FT}" = "true" ]; then
            echo ">>> Merging finetune LoRA weights for testing..."
            if [ "${LORA_ENABLE_PT}" = "true" ]; then
                FT_BASE_MODEL=${MERGED_OUTPUT_DIR}
            else
                FT_BASE_MODEL=${OUTPUT_DIR}/pretrain
                if [ ! -d "${FT_BASE_MODEL}" ]; then
                    FT_BASE_MODEL=${STUDENT_MODEL}
                fi
            fi
            MERGED_FT_DIR=${OUTPUT_DIR}/finetune_merged
            python ${WORK_DIR}/scripts/merge_lora_qwen3_vl.py \
                ${FT_BASE_MODEL} \
                ${FINETUNE_OUTPUT_DIR} \
                ${MERGED_FT_DIR}
            FINETUNE_OUTPUT_DIR=${MERGED_FT_DIR}
        fi
        
        bash $0 test ${FINETUNE_OUTPUT_DIR}
        
        echo "Full pipeline completed. Output saved to: ${OUTPUT_DIR}"
        ;;
    
    pretrain)
        validate_training_inputs
        echo ">>> Start Pre-training ..."
        cd ${WORK_DIR}
        
        if [ -n "$2" ]; then
            OUTPUT_DIR=$2/pretrain
        else
            OUTPUT_DIR=${OUTPUT_BASE_DIR}/outputs/kve_kd_qwen3_vl_${TIMESTAMP}/pretrain
        fi
        mkdir -p ${OUTPUT_DIR}
        
        (
            unset CUDA_VISIBLE_DEVICES
            deepspeed \
                --include ${DS_INCLUDE} \
                --master_port ${MASTER_PORT:-29500} \
                kve_kd/training/train.py \
                --deepspeed ${DEEPSPEED_CONFIG_PT} \
                --model_name_or_path ${STUDENT_MODEL} \
                $(if [ -n "${TEACHER_MODEL}" ]; then echo "--teacher_model_path ${TEACHER_MODEL}"; fi) \
                --data_path ${DATA_PATH}/pretrain_data/${PRETRAIN_DATA_JSON} \
                --image_folder ${DATA_PATH}/pretrain_data \
                --output_dir ${OUTPUT_DIR} \
                --distill ${DISTILL} \
                --model_type qwen3_vl \
                --task pretrain \
                --bf16 True \
                --num_train_epochs 1 \
                --per_device_train_batch_size ${PRETRAIN_BS} \
                --per_device_eval_batch_size ${PRETRAIN_BS} \
                --gradient_accumulation_steps ${PRETRAIN_GAS} \
                --eval_strategy "no" \
                --save_strategy $([ "${SAVE_CHECKPOINT}" = "true" ] && echo "steps" || echo "no") \
                --save_steps ${SAVE_STEPS} \
                --save_total_limit ${SAVE_TOTAL_LIMIT} \
                --learning_rate ${PRETRAIN_LR} \
                --weight_decay 0. \
                --warmup_ratio ${WARMUP_RATIO} \
                --lr_scheduler_type cosine \
                --logging_steps 5 \
                --model_max_length ${MODEL_MAX_LENGTH} \
                --min_pixels ${MIN_PIXELS} \
                --max_pixels ${MAX_PIXELS} \
                --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
                --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
                --dataloader_drop_last ${DATALOADER_DROP_LAST} \
                --report_to none \
                --lambda_rkld ${LAMBDA_RKLD} \
                --lambda_v_all ${LAMBDA_V_ALL} \
                --lambda_v_focus ${LAMBDA_V_FOCUS} \
                --lambda_attn ${LAMBDA_ATTN} \
                $([ "${ADAPTIVE_TOPK}" = "true" ] && echo "--adaptive_topk" || echo "") \
                --topk_fixed ${TOPK_FIXED} \
                --topk_min ${TOPK_MIN} \
                --topk_max ${TOPK_MAX} \
                --topk_gamma ${TOPK_GAMMA} \
                --topk_attn_layer_idx ${TOPK_ATTN_LAYER_IDX} \
                $([ "${USE_ENTROPY_WEIGHTING}" = "true" ] && echo "--use_entropy_weighting" || echo "") \
                $([ "${ENABLE_MIXED_ATTN}" = "true" ] && echo "--enable_mixed_attn" || echo "") \
                --key_vis_token_method ${KEY_VIS_TOKEN_METHOD} \
                --key_vis_p ${KEY_VIS_P} \
                --key_vis_layer_threshold ${KEY_VIS_LAYER_THRESHOLD} \
                --key_vis_tokens_threshold ${KEY_VIS_TOKENS_THRESHOLD} \
                $([ "${KEY_VIS_DEBUG}" = "true" ] && echo "--key_vis_debug" || echo "") \
                --key_vis_trigger_metric ${KEY_VIS_TRIGGER_METRIC} \
                --key_vis_rel_l2_layer_threshold ${KEY_VIS_REL_L2_LAYER_THRESHOLD} \
                --key_vis_trigger_scan_start_layer ${KEY_VIS_TRIGGER_SCAN_START_LAYER} \
                --key_vis_kve_query_mode ${KEY_VIS_KVE_QUERY_MODE} \
                $([ "${LORA_ENABLE}" = "true" ] && echo "--lora_enable" || echo "") \
                --lora_r ${LORA_R} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --lora_bias ${LORA_BIAS} \
                --lora_target_modules ${LORA_TARGET_MODULES} \
                --random_seed ${RANDOM_SEED} \
                $([ -n "${MM_PROJECTOR_LR_PT}" ] && echo "--mm_projector_lr ${MM_PROJECTOR_LR_PT}" || echo "") \
                2>&1 | tee ${OUTPUT_DIR}/log.txt
        )
        echo "Done."
        ;;
    
    finetune)
        cd ${WORK_DIR}
        
        if [ -n "$2" ]; then
            PRETRAIN_OUTPUT_DIR=$2/pretrain
            OUTPUT_DIR=$2/finetune
            if [ -z "${STUDENT_MODEL_FROM_PIPELINE}" ] && [ -d "${PRETRAIN_OUTPUT_DIR}" ]; then
                STUDENT_MODEL=${PRETRAIN_OUTPUT_DIR}
            fi
        else
            OUTPUT_DIR=${OUTPUT_BASE_DIR}/outputs/kve_kd_qwen3_vl_${TIMESTAMP}/finetune
        fi
        validate_training_inputs
        echo ">>> Start Fine-tuning ..."
        mkdir -p ${OUTPUT_DIR}
        
        (
            unset CUDA_VISIBLE_DEVICES
            deepspeed \
                --include ${DS_INCLUDE} \
                --master_port ${MASTER_PORT:-29500} \
                kve_kd/training/train.py \
                --deepspeed ${DEEPSPEED_CONFIG_FT} \
                --model_name_or_path ${STUDENT_MODEL} \
                $(if [ -n "${TEACHER_MODEL}" ]; then echo "--teacher_model_path ${TEACHER_MODEL}"; fi) \
                --data_path ${DATA_PATH}/finetune_data/${FINETUNE_DATA_JSON} \
                --image_folder ${DATA_PATH}/finetune_data \
                --output_dir ${OUTPUT_DIR} \
                --distill ${DISTILL} \
                --model_type qwen3_vl \
                --task finetune \
                --bf16 True \
                --num_train_epochs 1 \
                --per_device_train_batch_size ${FINETUNE_BS} \
                --per_device_eval_batch_size ${FINETUNE_BS} \
                --gradient_accumulation_steps ${FINETUNE_GAS} \
                --eval_strategy "no" \
                --save_strategy $([ "${SAVE_CHECKPOINT}" = "true" ] && echo "steps" || echo "no") \
                --save_steps ${SAVE_STEPS} \
                --save_total_limit ${SAVE_TOTAL_LIMIT_FT} \
                --learning_rate ${FINETUNE_LR} \
                --weight_decay 0. \
                --warmup_ratio ${WARMUP_RATIO} \
                --lr_scheduler_type cosine \
                --logging_steps 5 \
                --model_max_length ${MODEL_MAX_LENGTH} \
                --min_pixels ${MIN_PIXELS} \
                --max_pixels ${MAX_PIXELS} \
                --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
                --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
                --dataloader_drop_last ${DATALOADER_DROP_LAST} \
                --group_by_modality_length ${GROUP_BY_MODALITY_LENGTH} \
                --report_to none \
                --lambda_rkld ${LAMBDA_RKLD} \
                --lambda_v_all ${LAMBDA_V_ALL} \
                --lambda_v_focus ${LAMBDA_V_FOCUS} \
                --lambda_attn ${LAMBDA_ATTN} \
                $([ "${ADAPTIVE_TOPK}" = "true" ] && echo "--adaptive_topk" || echo "") \
                --topk_fixed ${TOPK_FIXED} \
                --topk_min ${TOPK_MIN} \
                --topk_max ${TOPK_MAX} \
                --topk_gamma ${TOPK_GAMMA} \
                --topk_attn_layer_idx ${TOPK_ATTN_LAYER_IDX} \
                $([ "${USE_ENTROPY_WEIGHTING}" = "true" ] && echo "--use_entropy_weighting" || echo "") \
                $([ "${ENABLE_MIXED_ATTN}" = "true" ] && echo "--enable_mixed_attn" || echo "") \
                --key_vis_token_method ${KEY_VIS_TOKEN_METHOD} \
                --key_vis_p ${KEY_VIS_P} \
                --key_vis_layer_threshold ${KEY_VIS_LAYER_THRESHOLD} \
                --key_vis_tokens_threshold ${KEY_VIS_TOKENS_THRESHOLD} \
                $([ "${KEY_VIS_DEBUG}" = "true" ] && echo "--key_vis_debug" || echo "") \
                --key_vis_trigger_metric ${KEY_VIS_TRIGGER_METRIC} \
                --key_vis_rel_l2_layer_threshold ${KEY_VIS_REL_L2_LAYER_THRESHOLD} \
                --key_vis_trigger_scan_start_layer ${KEY_VIS_TRIGGER_SCAN_START_LAYER} \
                --key_vis_kve_query_mode ${KEY_VIS_KVE_QUERY_MODE} \
                $([ "${LORA_ENABLE}" = "true" ] && echo "--lora_enable" || echo "") \
                --lora_r ${LORA_R} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --lora_bias ${LORA_BIAS} \
                --lora_target_modules ${LORA_TARGET_MODULES} \
                --random_seed ${RANDOM_SEED} \
                $([ -n "${MM_PROJECTOR_LR_FT}" ] && echo "--mm_projector_lr ${MM_PROJECTOR_LR_FT}" || echo "") \
                $([ -n "${RESUME_FROM_CHECKPOINT}" ] && echo "--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}" || echo "") \
                2>&1 | tee -a ${OUTPUT_DIR}/log.txt
        )
        echo "Done."
        ;;
    
    test)
        echo ">>> Start Evaluation ..."
        MODEL_PATH=$2
        EVAL_DATA_PATH=${3:-${DATA_PATH}}
        EVAL_OUTPUT_DIR=$4
        
        if [ -z "${MODEL_PATH}" ]; then
            echo "Error: Model path is required for test task"
            echo "Usage: $0 test <model_path> [data_path] [output_dir]"
            exit 1
        fi
        
        if [ -z "${EVAL_DATA_PATH}" ]; then
            echo "Error: Data path is required. Set DATA_PATH env or pass as argument."
            echo "Usage: $0 test <model_path> <data_path> [output_dir]"
            exit 1
        fi
        
        bash ${WORK_DIR}/scripts/benchmark.sh ${MODEL_PATH} ${EVAL_DATA_PATH} "${EVAL_OUTPUT_DIR}" qwen3_vl
        
        echo "Evaluation completed."
        ;;
    
    *)
        echo "Error: unknown task '${TASK}'" >&2
        echo "Valid tasks: pretrain, finetune, pretrain-finetune, pretrain-finetune-test, test" >&2
        exit 1
        ;;
esac
