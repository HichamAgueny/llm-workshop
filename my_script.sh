#!/bin/bash -e
# Set working directory and paths
PROJECT_NBR=nn9997k
MyWD="/cluster/work/projects/${PROJECT_NBR}/${USER}/llm-workshop"
CONTAINER_DIR="${MyWD}/container"
INPUT_DIR="${MyWD}/data"
TOOLS_DIR="${MyWD}/tools"
TOOLS_BIN="${TOOLS_DIR}/bin"

# Ensure required directories exist
mkdir -p "${INPUT_DIR}" "${CONTAINER_DIR}" "${TOOLS_BIN}"

echo "--Start copying the singularity image and base model"

# Copy base image to your work area
cp "/cluster/work/projects/${PROJECT_NBR}/hicham/llm-workshop/container/pytorch2.5_cu2.6.1_py3.10_baseimage_arm.sif" "${CONTAINER_DIR}/"

# Copy customized apptainer image to your work area
cp "/cluster/work/projects/${PROJECT_NBR}/hicham/llm-workshop/container/pytorch2.5_cu2.6.1_py3.10_custom.sif" "${CONTAINER_DIR}/"

# Modify the path in the .def file
cd "${CONTAINER_DIR}"
sed -i "s|/cluster/work/projects/${PROJECT_NBR}/hicham/llm-workshop/container|${CONTAINER_DIR}|g" "${CONTAINER_DIR}/pytorch2.5_cu2.6.1_py3.10_arm.def"

# Copy the base model to your work area
cp -r "/cluster/work/projects/${PROJECT_NBR}/hicham/llm-workshop/data/Llama-3.2-1B-Instruct" "${INPUT_DIR}/"

# Copy prompts folder
cp -r "/cluster/work/projects/${PROJECT_NBR}/hicham/llm-workshop/data/prompts" "${INPUT_DIR}/"

# Copy Xsum dataset
cp -r "/cluster/work/projects/${PROJECT_NBR}/hicham/llm-workshop/data/XSum" "${INPUT_DIR}/"

# Copy scripts from tools to tools/bin
chmod +x "${TOOLS_DIR}/monitor_singleGPU.sh"
chmod +x "${TOOLS_DIR}/monitor_multiGPU.sh"
cp "${TOOLS_DIR}/monitor_singleGPU.sh" "${TOOLS_BIN}/monitor_singleGPU"
cp "${TOOLS_DIR}/monitor_multiGPU.sh" "${TOOLS_BIN}/monitor_multiGPU"
chmod +x "${TOOLS_BIN}/monitor_singleGPU" "${TOOLS_BIN}/monitor_multiGPU"

# Define the line to be added to .bashrc (fully expanded path; escape $PATH so it expands on shell init)
PATH_LINE="export PATH=\"${TOOLS_BIN}:\$PATH\""
# Check if the line already exists in the .bashrc file to avoid duplicates.
if ! grep -qxF "$PATH_LINE" "$HOME/.bashrc"; then
    echo "$PATH_LINE" >> "$HOME/.bashrc"
    echo "Path added to .bashrc."
else
    echo "Path already exists in .bashrc. No changes made."
fi

# Make PATH available for the remainder of this script run
export PATH="${TOOLS_BIN}:$PATH"
# If the script is sourced, update the current shell too
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    source "$HOME/.bashrc"
fi

echo
echo "--Start updaing config. files"
# Update the path in your config. files
# Fine-tuning on a single GPU
FINETUNE_DIR="${MyWD}/fine-tuning-singlegpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/$PROJECT_NBR/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_single_device_QA.yaml"

# Fine-tuning on a multiple GPUs
FINETUNE_DIR="${MyWD}/fine-tuning-multigpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/$PROJECT_NBR/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_multi_device_QA.yaml"

# Exercise
FINETUNE_DIR="${MyWD}/exercise/fine-tuning-singlegpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/$PROJECT_NBR/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_single_device_Xsum.yaml"

# Fine-tuning on a multiple GPUs
FINETUNE_DIR="${MyWD}/exercise/fine-tuning-multigpu"
CONFIG_DIR="${FINETUNE_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/$PROJECT_NBR/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_multi_device_Xsum.yaml"

# Profiling
PROFILING_DIR="${MyWD}/profiling"
CONFIG_DIR="${PROFILING_DIR}/config_scripts"

cd ${CONFIG_DIR}
sed -i "s|/cluster/projects/$PROJECT_NBR/hicham/llm-workshop/data|$MyWD/data|g" "$CONFIG_DIR/1B_lora_single_device_QA_profiling.yaml"

echo "---finished :)"

exit
