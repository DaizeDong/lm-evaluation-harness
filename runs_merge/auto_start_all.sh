############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

task_name_list=("arc_challenge" "arc_easy" "boolq" "hellaswag" "lambada" "sciq")
#task_name_list=("arc_challenge" "arc_easy" "boolq" "hellaswag" "lambada" "logiqa" "mmlu" "nq_open" "piqa" "sciq" "winogrande")

for task_name in "${task_name_list[@]}"; do

  #  folder_name="2_8_layer0-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29_4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4E"
  #  folder_name="2_8_layer0-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29_8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8E"
  folder_name="2_8_layer0-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29_28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28E"

  #  folder_name="2_8_layer0-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29_28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28-28E"
  #  folder_name="2_8_layer0-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29_8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8-8E"
  #  folder_name="2_8_layer0-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29_4-6-8-8-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-4-8E"

  model_path="/mnt/petrelfs/share_data/quxiaoye/models/llama-moe-models/merged/${folder_name}"
  save_path="${root_dir}/results_merge/average/${task_name}/${folder_name}"

  #  model_path="/mnt/petrelfs/share_data/quxiaoye/models/llama-moe-models/merged_perm/${folder_name}"
  #  save_path="${root_dir}/results_merge/perm/${task_name}/${folder_name}"

  max_length=4096

  sbatch ${root_dir}/runs_merge/sub_tasks/${task_name}.sh ${model_path} ${save_path} ${max_length} &
  sleep 1
done
wait
