#!/bin/bash -e
#SBATCH --job-name=inference_llama3_1B_Xsum
#SBATCH --account=nn9997k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-cpu=8G
##SBATCH --nodelist=x1000c2s2b0n0

# Set proxy settings for HTTP and HTTPS traffic
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

echo "--Node: $(hostname)"
echo

# --- Variables and Paths ---
# Set working directory and paths
MyWD="/cluster/projects/nn9997k/$USER/llm-workshop"
INFERENCE_DIR="${MyWD}/inference"
CONTAINER_DIR="${MyWD}/container"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch2.5_cu2.6.1_py3.10_custom.sif"

CONFIG_DIR="${INFERENCE_DIR}/config_scripts"
PYTHON_DIR="${INFERENCE_DIR}/python_scripts"

# Set the path to the Python script that performs merging LoRA weight with base model & inference 
PYTHON_FILE="${PYTHON_DIR}/inference_peft_lora.py"

# Xsum task
# Define the base model and lora adpater paths for a specific task for merging... 
# Set the path to the directory containing the base model
BASE_MODEL_PATH="$MyWD/data/Llama-3.2-1B-Instruct"

# Set the path to the directory containing the saved LORA adapter weights
# LoRA weights generated from fine-tuning on a single GPU
LORA_ADAPTER_PATH="$MyWD/data/Llama-3.2-1B-Instruct_Xsum_out_OnlyLoRAweight"

# uncomment this for LoRA weights generated from fine-tuning on multiple GPUs
#LORA_ADAPTER_PATH="$MyWD/data/Llama-3.2-1B-Instruct_Xsum_out_4GPU_onlyLoRAweight"

# Set the path to the text file containing the prompts for the inference task
PROMPT_FILE="$MyWD/data/prompts/prompt_Xsum.txt"

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- My Directory: ${MyWD}"
echo "--- My FineTune Directory: ${INFERENCE_DIR}"
echo "--- My Container Directory: ${CONTAINER_DIR}"
echo "--- My based model path: ${BASE_MODEL_PATH}"
echo "--- My lora adapter path: ${LORA_ADAPTER_PATH}"
echo "--- My prompt file path: ${PROMPT_FILE}"
echo

# --- Create the Inner Script ---
# Use a temporary file for the inner script to avoid conflicts and ensure atomicity.
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"

cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

echo "Running Inference command:"
#tune run generate --config "${CONFIG_FILE}"
python "${PYTHON_FILE}" --base_model_path "${BASE_MODEL_PATH}" --lora_adapter_path "${LORA_ADAPTER_PATH}" --prompt_file "${PROMPT_FILE}"
EOF

chmod +x "${INNER_SCRIPT_TEMP}"

# --- Suppress LMOD Debugging ---
export LMOD_SH_DBG_ON=0

echo
echo "--- Launching the application ---"

# --- Execute with Apptainer ---
# Ensure -B bindings are correct. 
# Pass the full path to the temporary script.
time srun apptainer exec --nv -B "${MyWD}:${MyWD}" \
      "${APPTAINER_SIF}" \
      "${INNER_SCRIPT_TEMP}"

# --- Clean Up Temporary Script ---
rm -f "${INNER_SCRIPT_TEMP}"

echo
echo "--- Finished :) ---"
