#!/bin/bash

# Call the squeue command to get the job information for the user
job_info=$(squeue -u dongdaize.d)

# Use awk to sum up the GPUs used by the jobs listed in the squeue output
total_gpus=$(echo "$job_info" | awk 'BEGIN { total=0 } /gpu/ { gsub("gpu:",""); total += $9 } END { print total }')

# Output the total number of used GPUs
echo "Total used GPUs: $total_gpus"