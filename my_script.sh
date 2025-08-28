#!/bin/bash -e

# Set working directory and paths
PROJECT_NBR=nn9997k
MyWD="/cluster/work/projects/$PROJECT_NBR/$USER/llm-workshop"
CONTAINER_DIR="${MyWD}/container"

INPUT_DIR="$MyWD/data"

mkdir $INPUT_DIR

echo "--Start copying the singularity image and base model"
# Copy base image to your work area
cp /cluster/work/projects/nn9997k/hicham/llm-workshop/container/pytorch2.5_cu2.6.1_py3.10_baseimage_arm.sif "${CONTAINER_DIR}"

# Copy customized apptainer image to your work area
cp /cluster/work/projects/nn9997k/hicham/llm-workshop/container/pytorch2.5_cu2.6.1_py3.10_custom.sif "${CONTAINER_DIR}"

# Modify the path in the .def file
cd ${CONTAINER_DIR} 
sed -i "s|/cluster/work/projects/nn9997k/hicham/llm-workshop/container|${CONTAINER_DIR}|g" "${CONTAINER_DIR}/pytorch2.5_cu2.6.1_py3.10_arm.def"

# Copy the base model to your work area
cp -r /cluster/work/projects/nn9997k/hicham/llm-workshop/data/Llama-3.2-1B-Instruct ${INPUT_DIR}

# Copy prompts folder
cp -r /cluster/work/projects/nn9997k/hicham/llm-workshop/data/prompts ${INPUT_DIR}

# Copy scripts from tools to $HOME/.local/bin
chmod +x /cluster/work/projects/nn9997k/$USER/llm-workshop/tools/monitor_singleGPU.sh 
chmod +x /cluster/work/projects/nn9997k/$USER/llm-workshop/tools/monitor_multiGPU.sh

cp /cluster/work/projects/nn9997k/$USER/llm-workshop/tools/monitor_singleGPU.sh $HOME/.local/bin/monitor_singleGPU
cp /cluster/work/projects/nn9997k/$USER/llm-workshop/tools/monitor_multiGPU.sh $HOME/.local/bin/monitor_multiGPU

echo
echo "--Start updaing config. files"
# Update the path in your config. files
# Fine-tuning on a single GPU
FINETUNE_DIR="${MyWD}/fine-tuning-singlegpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/nn9997k/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_single_device_QA.yaml"

# Fine-tuning on a multiple GPUs
FINETUNE_DIR="${MyWD}/fine-tuning-multigpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/nn9997k/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_multi_device_QA.yaml"

# Exercise
FINETUNE_DIR="${MyWD}/exercise/fine-tuning-singlegpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/nn9997k/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_single_device_Xsum.yaml"

# Fine-tuning on a multiple GPUs
FINETUNE_DIR="${MyWD}/exercise/fine-tuning-multigpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/nn9997k/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_multi_device_Xsum.yaml"

# Profiling
PROFILING_DIR="${MyWD}/profiling"
CONFIG_DIR="${PROFILING_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/nn9997k/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_single_device_QA_profiling.yaml"

echo "---finished :)"

exit
