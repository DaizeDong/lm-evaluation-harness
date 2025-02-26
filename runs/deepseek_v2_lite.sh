#!/bin/bash

#export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,magnitude_l1,magnitude_l2"
#base_save_dir="/checkpoints/private/experiments/visualization"

export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,magnitude_l1,magnitude_l2,router_inputs"
base_save_dir="/dev/shm"

##########################################################################
model_path="/checkpoints/private/huggingface/DeepSeek-V2-lite"
output_path="${base_save_dir}/results/deepseek-v2-lite"
tensor_parallel_size=8
data_parallel_size=1
gpu_memory_utilization=0.8
max_model_len=4096
export ANALYSIS_SAVE_DIR="${base_save_dir}/deepseek-v2-lite"
export ENVIRON_SAVE_DIR=${ANALYSIS_SAVE_DIR}

lm_eval \
  --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tensor_parallel_size},data_parallel_size=${data_parallel_size},gpu_memory_utilization=${gpu_memory_utilization},max_model_len=${max_model_len},max_num_seqs=${max_model_len},trust_remote_code=True,dtype=auto,enforce_eager=True" \
  --output_path ${output_path} \
  --tasks winogrande \
  --batch_size 1
