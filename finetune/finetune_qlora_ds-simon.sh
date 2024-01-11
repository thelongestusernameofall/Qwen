#!/bin/bash

export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=ERROR

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

#export env_bin_path=/root/anaconda3/envs/qwen/bin/

#export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="/mnt/data/models/qwen/Qwen/Qwen-72B-Chat-Int4/"
DATA="/mnt/data/datasets/simon/sft/wanshitong/240109-alarm-sample-limit8000.json"
output_path="/mnt/data/models/qwen/Qwen/Qwen-72B-Chat-Int4-inf240109-v1-lora"

epochs=5
lr=3e-4
model_max_length=4096
lazy_preprocess=True
save_steps=3000
device_batch_size=1
gradient_accumulation_steps=8

lora_target_modules="c_attn, c_proj, w1, w2"

DS_CONFIG_PATH="finetune/ds_config_zero2.json"

function usage() {
    echo '
Usage: bash finetune/finetune_qlora_ds.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done


# Remember to use --fp16 instead of --bf16 due to autogptq
${env_bin_path}deepspeed --hostfile=/mnt/data/run/hostfile finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
    --output_dir ${output_path} \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${device_batch_size} \
    --per_device_eval_batch_size ${device_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length ${model_max_length} \
    --lazy_preprocess ${lazy_preprocess} \
    --use_lora \
    --q_lora \
    --gradient_checkpointing \
    --lora_target_modules ${lora_target_modules} \
    --deepspeed ${DS_CONFIG_PATH}