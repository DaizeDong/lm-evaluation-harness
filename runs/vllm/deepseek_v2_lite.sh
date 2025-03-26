#!/bin/bash

model_path="deepseek-ai/DeepSeek-V2-Lite"

export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,norm_weights,magnitude_l1,magnitude_l2"
save_dir=${model_path}
#export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,norm_weights,magnitude_l1,magnitude_l2,router_inputs"
#save_dir="/dev/shm"
export OVERWRITE_ANALYSIS_DATA="1"
export ANALYSIS_SAVE_DIR="${save_dir}/analysis/winogrande"
export ENVIRON_SAVE_DIR="${ANALYSIS_SAVE_DIR}/$(date +%Y%m%d-%H%M%S)"

##########################################################################
output_path="${save_dir}/results"
tensor_parallel_size=2
data_parallel_size=4
gpu_memory_utilization=0.8
max_model_len=4096

HF_ALLOW_CODE_EVAL="1" lm_eval \
  --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tensor_parallel_size},data_parallel_size=${data_parallel_size},gpu_memory_utilization=${gpu_memory_utilization},max_model_len=${max_model_len},max_num_seqs=${max_model_len},trust_remote_code=True,dtype=auto,enforce_eager=True" \
  --output_path ${output_path} \
  --tasks winogrande \
  --batch_size 1 \
  --trust_remote_code \
  --confirm_run_unsafe_code
