#!/bin/bash

## WARNING: Not working

use_tutel=""
#use_tutel="--use-tutel"

ds_inference=""
#ds_inference="--ds-inference"

export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_MODEL=/mnt/data/duarte/megatron-llama-2/7B-tp1-pp1
MOUNTED_MEGATRON_MODEL=/workspace/megatron-llama-2/7B-tp1-pp1
TOKENIZER=$MOUNTED_MEGATRON_MODEL/tokenizer.model

launch_cmd="deepspeed --num_nodes 1 --num_gpus 1"

program_cmd="tools/generate_samples_gpt.py \
    --tensor-model-parallel-size 1  \
    --pipeline-model-parallel-size 1  \
    --num-layers 32  \
    --hidden-size 4096  \
    --load ${MOUNTED_MEGATRON_MODEL}  \
    --max-position-embeddings 4096 \
    --num-attention-heads 32  \
    --ffn-hidden-size 11008  \
    --no-query-key-layer-scaling \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --bf16  \
    --micro-batch-size 1  \
    --seq-length 4096  \
    --out-seq-length 4096  \
    --temperature 1.0  \
    --tokenizer-type Llama2Tokenizer  \
    --tokenizer-model ${TOKENIZER}  \
    --top_p 0.9  \
    --seed 42 \
    --num-samples 1 \
    $use_tutel $ds_inference"

repo="/workspace/Megatron-Deepspeed"

echo $launch_cmd $program_cmd

docker run --gpus device=0 -it --rm --shm-size=128gb -p 5000:5000 \
    -v "$MEGATRON_MODEL:$MOUNTED_MEGATRON_MODEL" \
    -v "$(pwd):$repo" \
    -e "CUDA_DEVICE_MAX_CONNECTIONS=1" \
    -it --rm \
    megatron-deepspeed:latest \
    $launch_cmd $program_cmd
