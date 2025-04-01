# standard analysis
export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,layernorm_weights,activation_magnitude_l1,weights_magnitude_l1"
export ANALYSIS_SAVE_DIR="XXXXXXXXXXXXX"
export OVERWRITE_ANALYSIS_DATA="1"
export ANALYSIS_ARGS="max_tokens=100000" # (optional) in case the number of tokens are too high

# analyse router inputs
export ANALYSIS_TYPE="input_ids,balance_loss,router_scores,router_weights,router_bias,layernorm_weights,activation_magnitude_l1,weights_magnitude_l1,router_inputs"
export ANALYSIS_SAVE_DIR="/dev/shm" # save the `router_inputs` to the memory as they are way too big
export OVERWRITE_ANALYSIS_DATA="1"
export ANALYSIS_ARGS="max_tokens=100000"

# in case want to save the environ for debugging
export ENVIRON_SAVE_DIR="XXXXXXXXXXXXX"
