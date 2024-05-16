############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

max_length=4096
use_fast_tokenizer=True

folder_name_list=(
  #  "results_prune/DeepSeek-layer_drop-discrete-drop14"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop16"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop18"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop20"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop22"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop24"
  #  "results_prune/DeepSeek-layer_drop-discrete-drop26"

  #  "results_prune/DeepSeek-block_drop-discrete-drop10"
  #  "results_prune/DeepSeek-block_drop-discrete-drop12"
  #  "results_prune/DeepSeek-block_drop-discrete-drop14"
  #  "results_prune/DeepSeek-block_drop-discrete-drop16"
  #  "results_prune/DeepSeek-block_drop-discrete-drop18"
  #  "results_prune/DeepSeek-block_drop-discrete-drop20"
  #  "results_prune/DeepSeek-block_drop-discrete-drop22"
  #  "results_prune/DeepSeek-block_drop-discrete-drop24"
  #  "results_prune/DeepSeek-block_drop-discrete-drop26"

  "results_prune/DeepSeek-layer_drop-discrete-drop1"
  "results_prune/DeepSeek-layer_drop-discrete-drop3"
  "results_prune/DeepSeek-layer_drop-discrete-drop5"
  "results_prune/DeepSeek-layer_drop-discrete-drop7"
  "results_prune/DeepSeek-layer_drop-discrete-drop9"
  "results_prune/DeepSeek-layer_drop-discrete-drop11"
  "results_prune/DeepSeek-layer_drop-discrete-drop13"
  "results_prune/DeepSeek-layer_drop-discrete-drop15"
  "results_prune/DeepSeek-layer_drop-discrete-drop17"
  "results_prune/DeepSeek-layer_drop-discrete-drop19"
  "results_prune/DeepSeek-layer_drop-discrete-drop21"
  "results_prune/DeepSeek-layer_drop-discrete-drop23"
  "results_prune/DeepSeek-layer_drop-discrete-drop25"
  "results_prune/DeepSeek-layer_drop-discrete-drop27"

  "results_prune/DeepSeek-block_drop-discrete-drop1"
  "results_prune/DeepSeek-block_drop-discrete-drop3"
  "results_prune/DeepSeek-block_drop-discrete-drop5"
  "results_prune/DeepSeek-block_drop-discrete-drop7"
  "results_prune/DeepSeek-block_drop-discrete-drop9"
  "results_prune/DeepSeek-block_drop-discrete-drop11"
  "results_prune/DeepSeek-block_drop-discrete-drop13"
  "results_prune/DeepSeek-block_drop-discrete-drop15"
  "results_prune/DeepSeek-block_drop-discrete-drop17"
  "results_prune/DeepSeek-block_drop-discrete-drop19"
  "results_prune/DeepSeek-block_drop-discrete-drop21"
  "results_prune/DeepSeek-block_drop-discrete-drop23"
  "results_prune/DeepSeek-block_drop-discrete-drop25"
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

num_fewshot_list=(0 0 0 0 0 0)
task_name_list=("boolq" "hellaswag" "mmlu" "openbookqa" "rte" "winogrande")

#num_fewshot_list=(5)
#task_name_list=("gsm8k")

#num_fewshot_list=(0)
#task_name_list=("openbookqa")

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
    rm ${save_path}/results.json

    sbatch ${root_dir}/runs_prune/sub_tasks_deepseek/${task_name}.sh ${model_path} ${save_path} ${max_length} ${num_fewshot} $autogptq $autoawq $use_fast_tokenizer $sparse_type &
    sleep 1
  done
  sleep 60
done
wait
