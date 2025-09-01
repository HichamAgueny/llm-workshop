#!/bin/bash -e
#SBATCH --job-name=profiling-ft_llama3_1B_1GPU
#SBATCH --account=nn9970k
#SBATCH --time=00:35:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-cpu=8G
##SBATCH --nodelist=x1000c0s4b0n0

echo "--Node: $(hostname)"
echo

# --- Variables and Paths ---
# Set working directory and paths
PROJECT_NBR=nn9970k
MyWD="/cluster/work/projects/$PROJECT_NBR/$USER/llm-workshop"
PROFILING_DIR="${MyWD}/profiling"
CONTAINER_DIR="${MyWD}/container"
APPTAINER_SIF="${CONTAINER_DIR}/pytorch2.5_cu2.6.1_py3.10_custom.sif"

CONFIG_DIR="${PROFILING_DIR}/config_scripts"
PYTHON_DIR="${PROFILING_DIR}/python_scripts"

# QA
CONFIG_FILE="${CONFIG_DIR}/1B_lora_single_device_QA_profiling.yaml"
PYTHON_FILE="${PYTHON_DIR}/lora_finetune_single_device.py"

# Define the output & logging directories for fine-tuning results
OUTPUT_DIR="$MyWD/data/profiling_outputs"
LOGGING_DIR="$MyWD/data/lora_finetune_output"

# Create OUTPUT_DIR if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

# Create LOGGING_DIR if it doesn't exist
if [ ! -d "$LOGGING_DIR" ]; then
  echo "Creating logging directory: $LOGGING_DIR"
  mkdir -p "$LOGGING_DIR"
fi

# --- Locale Settings ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "--- My Main Directory: ${MyWD}"
echo "--- My FineTune Directory: ${FINETUNE_DIR}"
echo "--- My Container Directory: ${CONTAINER_DIR}"
echo "--- My Config-Files Directory: ${CONFIG_DIR}"
echo "--- My Config-Files Directory: ${PROFILING_DIR}"
echo

# --- Create the Inner Script ---
# Use a temporary file for the inner script to avoid conflicts and ensure atomicity.
INNER_SCRIPT_TEMP="./.my_script_temp_${SLURM_JOB_ID}"

cat > "${INNER_SCRIPT_TEMP}" << EOF
#!/bin/bash -e

# Flash Attention for efficiency
export USE_FLASH_ATTENTION=1

echo "Running fine-tuning command:"

# Example of overriding output (comment if NOT needed)
python "${PYTHON_FILE}" --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1

# Default usage (no overrides) 
#python "${PYTHON_FILE}" --config "${CONFIG_FILE}"
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
echo
rm -f "${INNER_SCRIPT_TEMP}"

echo
echo "--- Finished :) ---"
