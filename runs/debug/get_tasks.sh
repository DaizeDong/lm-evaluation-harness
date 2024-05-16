output_dir="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github"
output_file="${output_dir}/task_list.log"

mkdir -p ${output_dir}

gpus=0
cpus=8
quotatype=auto # auto spot reserved
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="test" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
  lm_eval --tasks list >${output_file} 2>&1 # Save both stdout and stderr to the file, overwriting if it exists
