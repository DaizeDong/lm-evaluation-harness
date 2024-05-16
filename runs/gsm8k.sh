#model_path=/mnt/petrelfs/share_data/quxiaoye/models_extra/tzhu_merge_candidates/phi-2-coder
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/phi-2-coder"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/tzhu_merge_candidates/phi-2-sft-dpo-gpt4_en-ep1"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/phi-2-sft-dpo-gpt4_en-ep1"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/tzhu_merge_candidates/phi-2-dpo-new"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/phi-2-dpo-new"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/tzhu_merge_candidates/dolphin-2_6-phi-2"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/dolphin-2_6-phi-2"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/tzhu_merge_candidates/phixtral-2x2_8"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/phixtral-2x2_8"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/tzhu_merge_candidates/phixtral-4x2_8"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/phixtral-4x2_8"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/microsoft--phi-2"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/microsoft--phi-2"

############################################################################################################

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/Llama_2_13b_chat"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/Llama_2_13b_chat"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/wizard_coder_python_13B_v10"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/wizard_coder_python_13B_v10"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/wizard_math_13B_v10"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/wizard_math_13B_v10"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/openchat_13B_v32_super"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/openchat_13B_v32_super"

#model_path="/mnt/petrelfs/share_data/quxiaoye/models_extra/vicuna_13B_v15"
#save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/vicuna_13B_v15"

model_path="/mnt/petrelfs/share_data/quxiaoye/models/wizard_lm_13B_v12"
save_path="/mnt/petrelfs/dongdaize.d/workspace-evaluaiton/lm-evaluation-harness-github/results/gsm8k/wizard_lm_13B_v12"

############################################################################################################

mkdir -p ${save_path}

gpus=8
cpus=$(($gpus * 16))
quotatype=auto # auto spot reserved
OMP_NUM_THREADS=4 srun --partition=MoE --job-name="eval" --mpi=pmi2 --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype} \
  accelerate launch -m lm_eval \
  --model hf \
  --model_args pretrained=${model_path},dtype="bfloat16",trust_remote_code=True,use_fast_tokenizer=False \
  --tasks gsm8k \
  --batch_size 8 \
  --log_samples \
  --output_path ${save_path}
