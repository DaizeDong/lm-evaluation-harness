#!/bin/bash

model_path="deepseek-ai/DeepSeek-V2"

export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,magnitude_l1,magnitude_l2"
save_dir=${model_path}
#export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,magnitude_l1,magnitude_l2,router_inputs"
#save_dir="/dev/shm"
export OVERWRITE_ANALYSIS_DATA="1"
export ANALYSIS_SAVE_DIR="${save_dir}/analysis"
export ANALYSIS_ARGS="save_interval_tokens=1000"
export ENVIRON_SAVE_DIR="${ANALYSIS_SAVE_DIR}/$(date +%Y%m%d-%H%M%S)"

##########################################################################
echo "Launching SGLang server...."

tp_size=8
mem_fraction_static=0.9

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NCCL_SOCKET_IFNAME="XXXXXXXXXXXX"
dist_init_addr="XXXXXXXXXXXX:XXXXXXXXXXXX"
nnodes=1
node_rank="XXXXXXXXXXXX"
host=127.0.0.1
port="XXXXXXXXXXXX"

python -m sglang.launch_server \
  --model-path ${model_path} \
  --tp-size ${tp_size} \
  --mem-fraction-static ${mem_fraction_static} \
  --dist-init-addr ${dist_init_addr} \
  --nnodes ${nnodes} \
  --node-rank ${node_rank} \
  --host ${host} \
  --port ${port} \
  --trust-remote-code &

##########################################################################
if [ ${node_rank} == 0 ]; then
  echo "Detecting server...."

  script_path="$(dirname "$(realpath "$0")")"
  python ${script_path}/wait_for_server.py \
    --host ${host} \
    --port ${port}

  echo "Launching LM-Harness...."

  max_length=4096
  output_path="${save_dir}/results"

  model_args="model=${model_path},base_url=http://${host}:${port}/v1/completions"
  model_args="${model_args},max_length=${max_length},tokenized_requests=False,trust_remote_code=True"

  lm_eval \
    --model local-completions \
    --model_args ${model_args} \
    --output_path ${output_path} \
    --tasks winogrande \
    --batch_size 1 \
    --trust_remote_code \
    --confirm_run_unsafe_code
else
  echo "Not launching LM-Harness, the rank (${node_rank}) is not 0."

  while true; do
    script_path="$(dirname "$(realpath "$0")")"
    python ${script_path}/connect_server.py \
      --host ${host} \
      --port ${port}

    sleep 3600
  done
fi
