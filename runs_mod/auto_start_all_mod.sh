############################################################################
root_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"

folder_name0="llama_mod"
parallelize=False

####################################################
folder_name1="converted"
#folder_name1="converted-cpt"

####################################################
#folder_name2="ablation_coefficient/vicuna_sharegpt-0.01"
#folder_name2="ablation_coefficient/vicuna_sharegpt-FULL-0.01"

#folder_name2="ablation_coefficient/vicuna_sharegpt-FULL-0.1"
#folder_name2="ablation_coefficient/evol_instruct-FULL-0.1"
#folder_name2="ablation_coefficient/slim_orca-FULL-0.1"
#folder_name2="ablation_coefficient/meta_math_qa-FULL-0.1"
#folder_name2="ablation_coefficient/evol_code_alpaca-FULL-0.1"

#folder_name2="ablation_coefficient/vicuna_sharegpt-FULL-1.0"
#folder_name2="ablation_coefficient/evol_instruct-FULL-1.0"
#folder_name2="ablation_coefficient/slim_orca-FULL-1.0"
#folder_name2="ablation_coefficient/meta_math_qa-FULL-1.0"
#folder_name2="ablation_coefficient/evol_code_alpaca-FULL-1.0"

#folder_name2="ablation_coefficient/mixed-0.1"
#folder_name2="ablation_coefficient/mixed-FULL-0.1"

#folder_name2="ablation_coefficient/mixed-1.0"
#folder_name2="ablation_coefficient/mixed-FULL-1.0"
#folder_name2="ablation_coefficient/mixed-FULL-1.0-cos0.1"

####################################################
#folder_name2="ablation_structure/mixed-FULL-cap0.5-cos1.0"
#folder_name2="ablation_structure/mixed-FULL-cap0.5-self1.0"
#folder_name2="ablation_structure/mixed-FULL-NoScale-cap0.5-cos1.0"
#folder_name2="ablation_structure/mixed-FULL-NoScale-freq4-cap0.5-cos1.0"
#folder_name2="ablation_structure/mixed-FULL-NoScale-freq2-cap0.5-cos1.0"

#folder_name2="ablation_structure/mixed-FULL-ExAttn-cap0.5-cos1.0"
#folder_name2="ablation_structure/mixed-FULL-ExAttn-cap0.5-self1.0"
#folder_name2="ablation_structure/mixed-FULL-ExAttn-NoScale-cap0.5-cos1.0"

####################################################
#folder_name2="ablation_capacity/mixed-FULL-ExAttn-NoScale-freq1-cap0.9-cos1.0"
#folder_name2="ablation_capacity/mixed-FULL-ExAttn-NoScale-freq1-cap0.8-cos1.0"

folder_name2="ablation_capacity/mixed-FULL-ExAttn-NoScale-freq1-cap0.9-cos-global1.0"
#folder_name2="ablation_capacity/mixed-FULL-ExAttn-NoScale-freq1-cap0.8-cos-global1.0"
#folder_name2="ablation_capacity/mixed-FULL-ExAttn-NoScale-freq1-cap0.7-cos-global1.0"
#folder_name2="ablation_capacity/mixed-FULL-ExAttn-NoScale-freq1-cap0.6-cos-global1.0"

#folder_name2="ablation_capacity/mixed-FULL-ExAttn-freq1-cap0.9-cos-global1.0"

####################################################

max_length=2048

num_fewshot_list=(25 10 5 0 5 5)
batch_size_list=(auto auto 32 64 auto 16)
task_name_list=("arc_challenge" "hellaswag" "mmlu" "truthfulqa" "winogrande" "gsm8k")

#num_fewshot_list=(0 5)
#batch_size_list=(64 64)
#task_name_list=("truthfulqa" "gsm8k")

#num_fewshot_list=(25 10 5 5)
#batch_size_list=(auto auto 32 auto)
#task_name_list=("arc_challenge" "hellaswag" "mmlu" "winogrande")

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
#batch_size_list=(16)
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
