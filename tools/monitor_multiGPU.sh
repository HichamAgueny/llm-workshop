#!/bin/bash

# Usage: ./monitor_multiGPU.sh <JOB_ID>
# To stop the script, press Ctrl+C

cp ./gpu_format.awk $HOME

JOB_ID="$1"

# Check for input argument
if [ -z "$1" ]; then
    echo "Usage: $0 <JobID>"
    exit 1
fi

# Check if job exists
if ! scontrol show job "$JOB_ID" > /dev/null 2>&1; then
    echo "Job ID $JOB_ID not found or no longer active."
    exit 1
fi

# Extract user and node
USER=$(scontrol show job "$JOB_ID" | awk -F= '/UserId/ {print $2}' | cut -d '(' -f1)
#NODE=$(scontrol show hostnames "$(scontrol show job $JOB_ID | awk -F= '/NodeList/ {print $2}' | cut -d ' ' -f1)" | head -n 1)
NODE=$(scontrol show job "$JOB_ID" | grep -oP 'NodeList=\K\S+' | grep -v '(null)' | head -n 1)

if [ -z "$NODE" ]; then
    echo "Failed to resolve a valid node for Job ID $JOB_ID"
    exit 1
fi

# Connect to node and run the monitoring inside job context
ssh -tt "$NODE" <<EOF
echo "Connected to node: $NODE"
echo "Starting interactive shell inside job $JOB_ID..."

srun --jobid=$JOB_ID --interactive --pty bash
echo "Inside job environment..."
echo 
echo "To stop the script, press Ctrl+C"

# Start monitoring specific GPU
watch -n 1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | awk -f ./gpu_format.awk"
EOF

