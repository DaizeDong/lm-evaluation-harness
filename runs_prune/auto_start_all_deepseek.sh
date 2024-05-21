############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

max_length=4096
use_fast_tokenizer=True

#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.1-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.2-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.3-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.4-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.5-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.6-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.7-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.8-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-4:8-0.5-128-NoAttn"
#folder_name="results_prune/DeepSeek-wanda-c4_train-2:4-0.5-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.1-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.2-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.3-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.4-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.5-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.6-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.7-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.8-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-4:8-0.5-128-NoAttn"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-2:4-0.5-128-NoAttn"

#folder_name="results_prune/DeepSeek-wanda-c4_train-unstructured-0.5-128-NoAttn-NoShared"
#folder_name="results_prune/DeepSeek-wanda-c4_train-4:8-0.5-128-NoAttn-NoShared"
#folder_name="results_prune/DeepSeek-wanda-c4_train-2:4-0.5-128-NoAttn-NoShared"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.5-128-NoAttn-NoShared"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-4:8-0.5-128-NoAttn-NoShared"
#folder_name="results_prune/DeepSeek-sparsegpt-c4_train-2:4-0.5-128-NoAttn-NoShared"

#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r0"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r8"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r16"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r24"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r32"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r40"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r48"
#folder_name="results_prune/DeepSeek-expert_drop-layerwise_pruning-r56"

#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r0"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r8"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r16"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r24"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r32"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r40"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r48"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r56"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r0-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r8-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r16-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r24-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r32-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r40-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r48-DyGate"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r56-DyGate"

#folder_name="results_prune/DeepSeek-layer_drop-consecutive-drop2"
#folder_name="results_prune/DeepSeek-layer_drop-consecutive-drop4"
#folder_name="results_prune/DeepSeek-layer_drop-consecutive-drop6"
#folder_name="results_prune/DeepSeek-layer_drop-consecutive-drop8"
#folder_name="results_prune/DeepSeek-layer_drop-consecutive-drop10"
#folder_name="results_prune/DeepSeek-layer_drop-consecutive-drop12"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop2"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop4"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop6"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop8"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop10"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop12"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop14"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop16"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop18"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop20"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop22"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop24"
#folder_name="results_prune/DeepSeek-layer_drop-discrete-drop26"

#folder_name="results_prune/DeepSeek-block_drop-consecutive-drop2"
#folder_name="results_prune/DeepSeek-block_drop-consecutive-drop4"
#folder_name="results_prune/DeepSeek-block_drop-consecutive-drop6"
#folder_name="results_prune/DeepSeek-block_drop-consecutive-drop8"
#folder_name="results_prune/DeepSeek-block_drop-consecutive-drop10"
#folder_name="results_prune/DeepSeek-block_drop-consecutive-drop12"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop2"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop4"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop6"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop8"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop10"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop12"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop14"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop16"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop18"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop20"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop22"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop24"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop26"

#folder_name="results_prune/DeepSeek-block_drop-discrete-drop2-expert_drop-global_pruning-r48"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop2-expert_drop-global_pruning-r56"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop4-expert_drop-global_pruning-r48"
#folder_name="results_prune/DeepSeek-block_drop-discrete-drop4-expert_drop-global_pruning-r56"

#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r48-block_drop-discrete-drop2"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r56-block_drop-discrete-drop2"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r48-block_drop-discrete-drop4"
#folder_name="results_prune/DeepSeek-expert_drop-global_pruning-r56-block_drop-discrete-drop4"

sparse_type="none"
autogptq=False
autoawq=False

#sparse_type="2:4"

# GPTQ
source ~/anaconda3/bin/activate awq
folder_name=results_quantization/deepseek-GPTQ-4bits
autogptq=True

# AWQ
#source ~/anaconda3/bin/activate awq
#folder_name="results_quantization/deepseek-AWQ-4bits"
#folder_name="results_assemble/deepseek-AWQ-4bits-block_drop-discrete-drop5"
#folder_name="results_assemble/deepseek-AWQ-4bits-expert_drop-global_pruning-r48"
#folder_name="results_assemble/deepseek-AWQ-4bits-layer_drop-discrete-drop4"
#autoawq=True

####################################################################
#num_fewshot_list=(5 0 0 0 0 0 0 0)
#task_name_list=("gsm8k" "arc_challenge" "boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(0 0 0 0 0 0 0 0)
#task_name_list=("piqa" "arc_challenge" "boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(0 0 0 0 0 0)
#task_name_list=("boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(0 0 0 0)
#task_name_list=("hellaswag" "mmlu" "openbookqa" "winogrande")

#num_fewshot_list=(0)
#task_name_list=("boolq")

#num_fewshot_list=(5)
#task_name_list=("gsm8k")

num_fewshot_list=(0)
task_name_list=("piqa")

# num_fewshot_list=(0)
# task_name_list=("winogrande")

for ((i = 0; i < ${#num_fewshot_list[@]}; i++)); do
  num_fewshot=${num_fewshot_list[i]}
  task_name=${task_name_list[i]}
  echo "${task_name}: ${num_fewshot} shot"
  echo "sparse_type: ${sparse_type}"

  ############## FOR ORIGINAL MODEL ##############
  #  model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
  #  save_path="${root_dir}/results_prune/${task_name}/${num_fewshot}shot-DeepSeek"
  #  sparse_type="none"
  #  autogptq=False
  #  autoawq=False

  ############## FOR COMPRESSED MODEL ##############
  model_path="/mnt/petrelfs/dongdaize.d/workspace/compression/${folder_name}/checkpoint"
  save_path="${root_dir}/results_prune/${task_name}/${num_fewshot}shot-${folder_name}"

  ##################################################
  result_file="${save_path}/results*.json"
  #  rm ${result_file}
  # if ls ${result_file} >/dev/null 2>&1; then
  #   echo "Result file \"${result_file}\" already exists. Do not apply the task."
  # else
  if [ ! -d ${model_path} ]; then
    echo "Model path \"${model_path}\" not exists. Do not apply the task."
  else
    sbatch ${root_dir}/runs_prune/sub_tasks_deepseek/${task_name}.sh ${model_path} ${save_path} ${max_length} ${num_fewshot} $autogptq $autoawq $use_fast_tokenizer $sparse_type
    sleep 1
  fi
  # fi
done
wait
