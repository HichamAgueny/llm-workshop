#!/bin/bash -e
#SBATCH --job-name=gpt4_Inference_llama3_1B
#SBATCH --account=nn9997k
#SBATCH --time=00:35:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-cpu=8G

# Modules
module load craype-accel-nvidia90

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

echo "--Node: $(hostname)"
echo

# --- Variables and Paths ---
# Set working directory and paths
MyWD="/cluster/projects/nn9997k/$USER/llm-workshop"
INFERENCE_DIR="${MyWD}/inference"
CONTAINER_DIR="${MyWD}/container"
APPTAINER_SIF="${CONTAINER_DIR}/PyTorch2.5_cu2.6.1_Py3.10.sif"
VENV_PATH="${CONTAINER_DIR}/VirtEnv"

CONFIG_DIR="${INFERENCE_DIR}/config_scripts"
PYTHON_DIR="${INFERENCE_DIR}/python_scripts"

# gpt4
CONFIG_FILE="${CONFIG_DIR}/llama_3.2_1B_generation_gpt4_QA.yaml"

PYTHON_FILE="${PYTHON_DIR}/generate.py"

# Define the output & logging directories for fine-tuning results
OUTPUT_DIR="$MyWD/data/Inference_results/Llama-3.2-1B-Instruct_inference_gpt4_out"

# Create OUTPUT_DIR if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- My Directory: ${MyWD}"
echo "--- My FineTune Directory: ${INFERENCE_DIR}"
echo "--- My Container Directory: ${CONTAINER_DIR}"
echo "--- My Scripts Directory: ${CONFIG_DIR}"
echo

# --- Create the Inner Script ---
# Use a temporary file for the inner script to avoid conflicts and ensure atomicity.
INNER_SCRIPT_TEMP="${CONFIG_DIR}/.my_script_temp_${SLURM_JOB_ID}"

cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

# Activate Virtual Environment
# Ensure the virtual environment is sourced correctly.
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "Error: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi

echo "Running Inference command:"
#tune run generate --config "${CONFIG_FILE}"
python "${PYTHON_FILE}" --config "${CONFIG_FILE}"
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
