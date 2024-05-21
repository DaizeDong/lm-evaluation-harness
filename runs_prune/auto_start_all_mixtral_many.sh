############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

max_length=4096

folder_name_list=(
  #  "results_prune/Mixtral-wanda-c4_train-unstructured-0.5-128-NoAttn"
  #  "results_prune/Mixtral-wanda-c4_train-4:8-0.5-128-NoAttn"
  #  "results_prune/Mixtral-wanda-c4_train-2:4-0.5-128-NoAttn"
  #  "results_prune/Mixtral-sparsegpt-c4_train-unstructured-0.5-128-NoAttn"
  #  "results_prune/Mixtral-sparsegpt-c4_train-4:8-0.5-128-NoAttn"
  #  "results_prune/Mixtral-sparsegpt-c4_train-2:4-0.5-128-NoAttn"

  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r0"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r1"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r2"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r3"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r4"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r5"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r6"
  #  "results_prune/Mixtral-expert_drop-layerwise_pruning-r7"

  #  "results_prune/Mixtral-expert_drop-global_pruning-r0"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r1"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r2"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r3"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r4"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r5"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r6"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r7"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r0-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r1-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r2-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r3-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r4-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r5-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r6-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r7-Reversed"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r0-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r1-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r2-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r3-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r4-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r5-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r6-DyGate"
  #  "results_prune/Mixtral-expert_drop-global_pruning-r7-DyGate"

  #  "results_prune/Mixtral-layer_drop-discrete-drop1"
  #  "results_prune/Mixtral-layer_drop-discrete-drop2"
  #  "results_prune/Mixtral-layer_drop-discrete-drop3"
  #  "results_prune/Mixtral-layer_drop-discrete-drop4"
  #  "results_prune/Mixtral-layer_drop-discrete-drop5"
  #  "results_prune/Mixtral-layer_drop-discrete-drop6"
  #  "results_prune/Mixtral-layer_drop-discrete-drop7"
  #  "results_prune/Mixtral-layer_drop-discrete-drop8"
  #  "results_prune/Mixtral-layer_drop-discrete-drop9"
  #  "results_prune/Mixtral-layer_drop-discrete-drop10"
  #  "results_prune/Mixtral-layer_drop-discrete-drop11"
  #  "results_prune/Mixtral-layer_drop-discrete-drop12"
  #  "results_prune/Mixtral-layer_drop-discrete-drop13"
  #  "results_prune/Mixtral-layer_drop-discrete-drop14"
  #  "results_prune/Mixtral-layer_drop-discrete-drop15"
  #  "results_prune/Mixtral-layer_drop-discrete-drop16"
  #  "results_prune/Mixtral-layer_drop-discrete-drop17"
  #  "results_prune/Mixtral-layer_drop-discrete-drop18"
  #  "results_prune/Mixtral-layer_drop-discrete-drop19"
  #  "results_prune/Mixtral-layer_drop-discrete-drop20"
  #  "results_prune/Mixtral-layer_drop-discrete-drop21"
  #  "results_prune/Mixtral-layer_drop-discrete-drop22"
  #  "results_prune/Mixtral-layer_drop-discrete-drop23"
  #  "results_prune/Mixtral-layer_drop-discrete-drop24"
  #  "results_prune/Mixtral-layer_drop-discrete-drop25"
  #  "results_prune/Mixtral-layer_drop-discrete-drop26"
  #  "results_prune/Mixtral-layer_drop-discrete-drop27"
  #  "results_prune/Mixtral-layer_drop-discrete-drop28"
  #  "results_prune/Mixtral-layer_drop-discrete-drop29"
  #  "results_prune/Mixtral-layer_drop-discrete-drop30"
  #  "results_prune/Mixtral-layer_drop-discrete-drop31"
  #  "results_prune/Mixtral-layer_drop-discrete-drop32"

  "results_prune/Mixtral-layer_drop-discrete-drop1-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop2-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop3-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop4-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop5-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop6-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop7-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop8-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop9-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop10-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop11-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop12-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop13-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop14-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop15-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop16-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop17-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop18-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop19-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop20-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop21-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop22-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop23-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop24-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop25-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop26-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop27-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop28-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop29-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop30-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop31-lima"
  "results_prune/Mixtral-layer_drop-discrete-drop32-lima"

  #  "results_prune/Mixtral-block_drop-discrete-drop1"
  #  "results_prune/Mixtral-block_drop-discrete-drop2"
  #  "results_prune/Mixtral-block_drop-discrete-drop3"
  #  "results_prune/Mixtral-block_drop-discrete-drop4"
  #  "results_prune/Mixtral-block_drop-discrete-drop5"
  #  "results_prune/Mixtral-block_drop-discrete-drop6"
  #  "results_prune/Mixtral-block_drop-discrete-drop7"
  #  "results_prune/Mixtral-block_drop-discrete-drop8"
  #  "results_prune/Mixtral-block_drop-discrete-drop9"
  #  "results_prune/Mixtral-block_drop-discrete-drop10"
  #  "results_prune/Mixtral-block_drop-discrete-drop11"
  #  "results_prune/Mixtral-block_drop-discrete-drop12"
  #  "results_prune/Mixtral-block_drop-discrete-drop13"
  #  "results_prune/Mixtral-block_drop-discrete-drop14"
  #  "results_prune/Mixtral-block_drop-discrete-drop15"
  #  "results_prune/Mixtral-block_drop-discrete-drop16"
  #  "results_prune/Mixtral-block_drop-discrete-drop17"
  #  "results_prune/Mixtral-block_drop-discrete-drop18"
  #  "results_prune/Mixtral-block_drop-discrete-drop19"
  #  "results_prune/Mixtral-block_drop-discrete-drop20"
  #  "results_prune/Mixtral-block_drop-discrete-drop21"
  #  "results_prune/Mixtral-block_drop-discrete-drop22"
  #  "results_prune/Mixtral-block_drop-discrete-drop23"
  #  "results_prune/Mixtral-block_drop-discrete-drop24"
  #  "results_prune/Mixtral-block_drop-discrete-drop25"
  #  "results_prune/Mixtral-block_drop-discrete-drop26"
  #  "results_prune/Mixtral-block_drop-discrete-drop27"
  #  "results_prune/Mixtral-block_drop-discrete-drop28"
  #  "results_prune/Mixtral-block_drop-discrete-drop29"
  #  "results_prune/Mixtral-block_drop-discrete-drop30"
  #  "results_prune/Mixtral-block_drop-discrete-drop31"
  #  "results_prune/Mixtral-block_drop-discrete-drop32"
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

#num_fewshot_list=(0)
#task_name_list=("boolq")

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
    #  model_path=/mnt/petrelfs/dongdaize.d/workspace/compression/models/mixtral
    #  save_path="${root_dir}/results_prune/${task_name}/${num_fewshot}shot-Mixtral"

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
        sbatch ${root_dir}/runs_prune/sub_tasks_mixtral/${task_name}.sh ${model_path} ${save_path} ${max_length} ${num_fewshot} $autogptq $autoawq $sparse_type &
        sleep 1
      fi
    fi
  done
done
wait
