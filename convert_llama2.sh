# This script will try to run a task *outside* any specified submitter
# Note: This script is for archival; it is not actually run by ducttape

model_size="7B"
hf_model="/mnt/data/duarte/hf-llama-2/$model_size"

tp="1"
pp="1"
megatron_model="/mnt/data/duarte/megatron-llama-2/$model_size-tp$tp-pp$pp"

repo="/workspace/Megatron-Deepspeed"

mount_hf_model="/workspace/hf-llama-2/$model_size"
mount_megatron_model="/workspace/megatron-llama-2/$model_size-tp$tp-pp$pp"

docker run --gpus 0 -it --rm --shm-size=128gb \
    -v "$hf_model:$mount_hf_model" \
    -v "$megatron_model:$mount_megatron_model" \
    megatron-deepspeed:latest \
    python $repo/tools/checkpoint_util.py \
        --model-type GPT \
        --loader llama2 \
        --saver megatron \
        --megatron-path $repo/megatron \
        --load-dir $mount_hf_model \
        --save-dir $mount_megatron_model \
        --tokenizer-model $mount_hf_model/tokenizer.model \
        --target-pipeline-parallel-size $pp \
        --target-tensor-parallel-size $tp

user=$(id -u):$(id -g)
echo "Changing ownership of $megatron_model to $user"
sudo chown -R $user $megatron_model

cp $hf_model/tokenizer.model $megatron_model/tokenizer.model