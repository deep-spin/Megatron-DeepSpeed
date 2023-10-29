#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

MEGATRON_MODEL=/mnt/data/duarte/megatron-llama-2/7B-tp1-pp1
MOUNTED_MEGATRON_MODEL=/workspace/megatron-llama-2/7B-tp1-pp1
TOKENIZER=$MOUNTED_MEGATRON_MODEL/tokenizer.model

export CUDA_DEVICE_MAX_CONNECTIONS=1

repo="/workspace/Megatron-Deepspeed"

docker run --gpus 0 -it --rm --shm-size=128gb -p 5000:5000 \
    -v "$MEGATRON_MODEL:$MOUNTED_MEGATRON_MODEL" \
    -v "$(pwd):$repo" \
    -e "CUDA_DEVICE_MAX_CONNECTIONS=1" \
    megatron-deepspeed:latest \
    torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
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
        --seed 42
