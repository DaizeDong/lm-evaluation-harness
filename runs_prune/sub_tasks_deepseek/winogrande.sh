#!/usr/bin/bash

#SBATCH --job-name=winogrande
#SBATCH --output=/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/logs_prune/winogrande/%x-%j.log
#SBATCH --error=/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/logs_prune/winogrande/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto
# reserved spot auto

num_nodes=1        # should match with --nodes
num_gpu_per_node=1 # should match with --gres
export OMP_NUM_THREADS=8
export LOGLEVEL=INFO

{
  # @Desc 此脚本用于获取一个指定区间且未被占用的随机端口号
  # @Author Hellxz <hellxz001@foxmail.com>

  function Listening { #判断当前端口是否被占用，没被占用返回0，反之1
    TCPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l)
    UDPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l)
    ((Listeningnum = TCPListeningnum + UDPListeningnum))
    if [ $Listeningnum == 0 ]; then
      echo "0"
    else
      echo "1"
    fi
  }

  function get_random_port { #得到随机端口
    PORT=0
    while [ $PORT == 0 ]; do
      temp_port=$(shuf -i $1-$2 -n1) #指定区间随机数
      if [ $(Listening $temp_port) == 0 ]; then
        PORT=$temp_port
      fi
    done
    echo "$PORT"
  }

  port=$(get_random_port 29500 29600) #任取一个未占用端口号
  echo "Port: $port"
}

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"
echo "Node list: $SLURM_JOB_NODELIS"

num_processes=$(expr ${num_nodes} \* ${num_gpu_per_node})
echo "Total Nodes: $num_nodes"
echo "Total GPUs: $num_processes"

############################################################################################################
model_path=$1
save_path=$2
max_length=$3
num_fewshot=$4
autogptq=$5
autoawq=$6
use_fast_tokenizer=$7
sparse_type=$8

trust_remote_code=True

if [ $autogptq == True ]; then
  trust_remote_code=False
fi

if [ $autoawq == True ]; then
  trust_remote_code=False
fi

echo "Model Path \"${model_path}\""
echo "Save Path \"${save_path}\""
echo "Sequence Length \"${max_length}\""
echo "trust_remote_code \"${trust_remote_code}\""

mkdir -p ${save_path}

srun lm_eval \
  --model hf \
  --model_args pretrained=${model_path},dtype="bfloat16",parallelize=True,trust_remote_code=${trust_remote_code},use_fast_tokenizer=${use_fast_tokenizer},max_length=${max_length},autogptq=${autogptq},autoawq=${autoawq},sparse_type=${sparse_type} \
  --tasks winogrande \
  --num_fewshot ${num_fewshot} \
  --batch_size "auto" \
  --output_path ${save_path}
