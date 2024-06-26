############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

max_length=4096
use_fast_tokenizer=True

folder_name_list=(
  #  "results_prune/DeepSeek-wanda-c4_train-unstructured-0.5-128-NoAttn"
  #  "results_prune/DeepSeek-wanda-c4_train-unstructured-0.5-128-NoAttn-NoShared"
  #  "results_prune/DeepSeek-wanda-c4_train-4:8-0.5-128-NoAttn"
  #  "results_prune/DeepSeek-wanda-c4_train-4:8-0.5-128-NoAttn-NoShared"
  #  "results_prune/DeepSeek-wanda-c4_train-2:4-0.5-128-NoAttn"
  #  "results_prune/DeepSeek-wanda-c4_train-2:4-0.5-128-NoAttn-NoShared"
  #  "results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.5-128-NoAttn"
  #  "results_prune/DeepSeek-sparsegpt-c4_train-unstructured-0.5-128-NoAttn-NoShared"
  #  "results_prune/DeepSeek-sparsegpt-c4_train-4:8-0.5-128-NoAttn"
  #  "results_prune/DeepSeek-sparsegpt-c4_train-4:8-0.5-128-NoAttn-NoShared"
  #  "results_prune/DeepSeek-sparsegpt-c4_train-2:4-0.5-128-NoAttn"
  #  "results_prune/DeepSeek-sparsegpt-c4_train-2:4-0.5-128-NoAttn-NoShared"

  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r0"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r4"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r8"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r12"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r16"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r20"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r24"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r28"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r32"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r36"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r40"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r44"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r48"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r52"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r56"
  #  "results_prune/DeepSeek-expert_drop-layerwise_pruning-r60"

  #  "results_prune/DeepSeek-expert_drop-global_pruning-r0"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r4"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r8"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r12"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r16"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r20"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r24"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r28"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r32"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r36"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r40"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r44"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r48"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r52"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r56"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r60"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r0-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r8-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r16-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r24-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r32-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r40-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r48-DyGate"
  #  "results_prune/DeepSeek-expert_drop-global_pruning-r56-DyGate"

  #  "results_prune/DeepSeek-layer_drop-discrete-drop1"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop2"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop3"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop4"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop5"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop6"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop7"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop8"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop9"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop10"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop11"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop12"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop13"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop14"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop15"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop16"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop17"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop18"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop19"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop20"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop21"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop22"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop23"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop24"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop25"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop26"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop27"

  "results_prune/DeepSeek-block_drop-discrete-drop1"
  "results_prune/DeepSeek-block_drop-discrete-drop2"
  "results_prune/DeepSeek-block_drop-discrete-drop3"
  "results_prune/DeepSeek-block_drop-discrete-drop4"
  "results_prune/DeepSeek-block_drop-discrete-drop5"
  "results_prune/DeepSeek-block_drop-discrete-drop6"
  "results_prune/DeepSeek-block_drop-discrete-drop7"
  "results_prune/DeepSeek-block_drop-discrete-drop8"
  "results_prune/DeepSeek-block_drop-discrete-drop9"
  "results_prune/DeepSeek-block_drop-discrete-drop10"
  "results_prune/DeepSeek-block_drop-discrete-drop11"
  "results_prune/DeepSeek-block_drop-discrete-drop12"
  "results_prune/DeepSeek-block_drop-discrete-drop13"
  "results_prune/DeepSeek-block_drop-discrete-drop14"
  "results_prune/DeepSeek-block_drop-discrete-drop15"
  "results_prune/DeepSeek-block_drop-discrete-drop16"
  "results_prune/DeepSeek-block_drop-discrete-drop17"
  "results_prune/DeepSeek-block_drop-discrete-drop18"
  "results_prune/DeepSeek-block_drop-discrete-drop19"
  "results_prune/DeepSeek-block_drop-discrete-drop20"
  "results_prune/DeepSeek-block_drop-discrete-drop21"
  "results_prune/DeepSeek-block_drop-discrete-drop22"
  "results_prune/DeepSeek-block_drop-discrete-drop23"
  "results_prune/DeepSeek-block_drop-discrete-drop24"
  "results_prune/DeepSeek-block_drop-discrete-drop25"
  "results_prune/DeepSeek-block_drop-discrete-drop26"
  "results_prune/DeepSeek-block_drop-discrete-drop27"
)

sparse_type="none"
autogptq=False
autoawq=False

#sparse_type="2:4"
#autogptq=True
#autoawq=True

####################################################################
#num_fewshot_list=(5 0 0 0 0 0 0 0 0)
#task_name_list=("gsm8k" "arc_challenge" "arc_easy" "boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(0 0 0 0 0 0 0 0)
#task_name_list=("triviaqa" "arc_challenge" "boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

num_fewshot_list=(0 0 0 0 0 0 0 0)
task_name_list=("piqa" "arc_challenge" "boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(0 0 0 0 0 0)
#task_name_list=("boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(0 0 0 0)
#task_name_list=("hellaswag" "mmlu" "openbookqa" "winogrande")

#num_fewshot_list=(0 0)
#task_name_list=("piqa" "arc_challenge")

num_fewshot_list=(0)
task_name_list=("rte")

#num_fewshot_list=(0)
#task_name_list=("piqa")

#num_fewshot_list=(0)
#task_name_list=("winogrande")

for folder_name in "${folder_name_list[@]}"; do
  for ((i = 0; i < ${#num_fewshot_list[@]}; i++)); do
    num_fewshot=${num_fewshot_list[i]}
    task_name=${task_name_list[i]}
    echo "${task_name}: ${num_fewshot} shot"
    echo "sparse_type: ${sparse_type}"

    ############## FOR ORIGINAL MODEL ##############
    #  model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/deepseek
    #  save_path="${root_dir}/results_prune/${task_name}/${num_fewshot}shot-DeepSeek"

    ############## FOR COMPRESSED MODEL ##############
    model_path="/mnt/petrelfs/dongdaize.d/workspace/compression/${folder_name}/checkpoint"
    save_path="${root_dir}/results_prune/${task_name}/${num_fewshot}shot-${folder_name}"

    ##################################################
    result_file="${save_path}/results*.json"
    #    rm ${result_file}
    if ls ${result_file} >/dev/null 2>&1; then
      echo "Result file \"${result_file}\" already exists. Do not apply the task."
    else
      if [ ! -d ${model_path} ]; then
        echo "Model path \"${model_path}\" not exists. Do not apply the task."
      else
        sbatch ${root_dir}/runs_prune/sub_tasks_deepseek/${task_name}.sh ${model_path} ${save_path} ${max_length} ${num_fewshot} $autogptq $autoawq $use_fast_tokenizer $sparse_type
        sleep 1
      fi
    fi
  done
done
wait
