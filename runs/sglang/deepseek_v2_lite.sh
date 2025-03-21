#!/bin/bash

model_path="deepseek-ai/DeepSeek-V2-Lite"

export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,magnitude_l1,magnitude_l2"
save_dir=${model_path}
#export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,magnitude_l1,magnitude_l2,router_inputs"
#save_dir="/dev/shm"
export OVERWRITE_ANALYSIS_DATA="1"
export ANALYSIS_SAVE_DIR="${save_dir}/analysis/winogrande"
export ANALYSIS_ARGS="save_interval_tokens=1000"
export ENVIRON_SAVE_DIR="${ANALYSIS_SAVE_DIR}/$(date +%Y%m%d-%H%M%S)"

##########################################################################
output_path="${save_dir}/results"
tp_size=1
dp_size=8
mem_fraction_static=0.8
max_model_len=4096

HF_ALLOW_CODE_EVAL="1" lm_eval \
  --model sglang \
  --model_args "pretrained=${model_path},dp_size=${dp_size},tp_size=${tp_size},mem_fraction_static=${mem_fraction_static},max_model_len=${max_model_len},trust_remote_code=True,dtype=auto" \
  --output_path ${output_path} \
  --tasks winogrande \
  --batch_size 1 \
  --trust_remote_code \
  --confirm_run_unsafe_code
