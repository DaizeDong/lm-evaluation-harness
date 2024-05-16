############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

folder_name0="llama_pro"
parallelize=True

##########################
folder_name1="converted"
#folder_name1="converted-cpt"

##########################
#folder_name2="mixed"
folder_name2="mixed-FULL"
#folder_name2="vicuna_sharegpt"
#folder_name2="vicuna_sharegpt-FULL"
#folder_name2="evol_instruct"
#folder_name2="evol_instruct-FULL"
#folder_name2="slim_orca"
#folder_name2="slim_orca-FULL"
#folder_name2="meta_math_qa"
#folder_name2="meta_math_qa-FULL"
#folder_name2="evol_code_alpaca"
#folder_name2="evol_code_alpaca-FULL"

##########################
max_length=4096

num_fewshot_list=(25 10 5 0 5 5)
batch_size_list=(auto auto 32 64 auto 64)
task_name_list=("arc_challenge" "hellaswag" "mmlu" "truthfulqa" "winogrande" "gsm8k")

#num_fewshot_list=(25)
#batch_size_list=(auto)
#task_name_list=("arc_challenge")

#num_fewshot_list=(10)
#batch_size_list=(auto)
#task_name_list=("hellaswag")

#num_fewshot_list=(5)
#batch_size_list=(32)
#task_name_list=("mmlu")

#num_fewshot_list=(0)
#batch_size_list=(64)
#task_name_list=("truthfulqa")

#num_fewshot_list=(5)
#batch_size_list=(64)
#task_name_list=("winogrande")

#num_fewshot_list=(5)
#batch_size_list=(64)
#task_name_list=("gsm8k")

for ((i = 0; i < ${#num_fewshot_list[@]}; i++)); do
  batch_size=${batch_size_list[i]}
  num_fewshot=${num_fewshot_list[i]}
  task_name=${task_name_list[i]}
  echo "${task_name}: ${num_fewshot} shot"

  model_path="/mnt/petrelfs/dongdaize.d/workspace/depth-llama/results/finetune/${folder_name0}/${folder_name1}/${folder_name2}"
  save_path="${root_dir}/results_mod/${task_name}-${num_fewshot}shot/${folder_name0}-${folder_name1}/${folder_name2}"
  rm ${save_path}/results.json

  sbatch ${root_dir}/runs_mod/sub_tasks/${task_name}.sh ${model_path} ${save_path} ${batch_size} ${max_length} ${num_fewshot} ${parallelize} &
  sleep 1
done
wait
