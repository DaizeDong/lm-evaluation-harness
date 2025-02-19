############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

folder_name0="mistral_mod"
parallelize=False

####################################################
folder_name1="paper"

####################################################
#folder_name2="main_results/baseline-mistral"
#folder_name2="main_results/mistral-mod-interleave-zeros32-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/mistral-mod-interleave-zeros32-freq1-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"

#folder_name2="main_results/baseline-mistral-pro"

#folder_name2="main_results/mistral-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="main_results/mistral-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"

####################################################
#folder_name2="alpaca/baseline-mistral"
#folder_name2="alpaca/mistral-mod-interleave-zeros32-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="alpaca/mistral-mod-interleave-zeros32-freq1-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"

#folder_name2="alpaca/baseline-mistral-pro"

folder_name2="alpaca/mistral-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.9-cos-global1.0-Anneal1000-DyLr"
#folder_name2="alpaca/mistral-mod-interleave-zeros40-freq1-Scale2.0-Gap0.05-freq1-cap0.8-cos-global1.0-Anneal1000-DyLr"

####################################################

max_length=4096

#num_fewshot_list=(0 0 0 0 -1 25 5 0 5 5)
#batch_size_list=(auto auto auto auto auto auto 32 64 auto 16)
#task_name_list=("mathqa" "piqa" "glue" "openbookqa" "squadv2" "arc_challenge" "mmlu" "truthfulqa" "winogrande" "gsm8k")

num_fewshot_list=(0 0 0 0 25 0 5)
batch_size_list=(auto auto auto auto auto 64 auto)
task_name_list=("mathqa" "piqa" "glue" "openbookqa" "arc_challenge" "truthfulqa" "winogrande")

for ((i = 0; i < ${#num_fewshot_list[@]}; i++)); do
  batch_size=${batch_size_list[i]}
  num_fewshot=${num_fewshot_list[i]}
  task_name=${task_name_list[i]}
  echo "${task_name}: ${num_fewshot} shot"

  #################### BASELINE ########################
  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/llama2_7B"
  #  save_path="${root_dir}/results_mod/${task_name}-${num_fewshot}shot/llama2_7B"

  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/LLaMA-Pro-8B"
  #  save_path="${root_dir}/results_mod/${task_name}-${num_fewshot}shot/LLaMA-Pro-8B"

  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/Mistral-7B-v0.1"
  #  save_path="${root_dir}/results_mod/${task_name}-${num_fewshot}shot/Mistral-7B-v0.1"

  #  model_path="/mnt/petrelfs/dongdaize.d/quxioaye/models/Mistral_Pro_8B_v0.1"
  #  save_path="${root_dir}/results_mod/${task_name}-${num_fewshot}shot/Mistral_Pro_8B_v0.1"

  ######################################################
  model_path="/mnt/petrelfs/dongdaize.d/workspace/depth-llama/results/finetune/${folder_name0}/${folder_name1}/${folder_name2}"
  save_path="${root_dir}/results_mod/${task_name}-${num_fewshot}shot/${folder_name0}-${folder_name1}/${folder_name2}"

  result_file="${save_path}/results*.json"
  #  rm ${save_path}/results.json
  if ls ${result_file} >/dev/null 2>&1; then
    echo "Result file \"${result_file}\" already exists. Do not apply the task."
  else
    if [ ! -d ${model_path} ]; then
      echo "Model path \"${model_path}\" not exists. Do not apply the task."
    else
      sbatch ${root_dir}/runs_mod/sub_tasks/${task_name}.sh ${model_path} ${save_path} ${batch_size} ${max_length} ${num_fewshot} ${parallelize} &
      sleep 1
    fi
  fi
done
wait
