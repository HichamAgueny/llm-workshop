#!/bin/bash -e
#SBATCH --job-name=1FineTune_llama3_11B_4GPU_srun-tunerun
#SBATCH --account=nn9997k
#SBATCH --time=2-20:35:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -o ./out/%x-%j.out
#SBATCH --mem-per-cpu=8G
#SBATCH --nodelist=x1000c1s6b0n0

# Modules
#module load craype-accel-nvidia90

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

echo "--Node: $(hostname)"
echo

# --- Variables and Paths ---
# Set working directory and paths
MyWD="/cluster/projects/nn9997k/$USER/llm-workshop"
FINETUNE_DIR="${MyWD}/exercise/fine-tuning-multigpu"
CONTAINER_DIR="${MyWD}/container"
APPTAINER_SIF="${CONTAINER_DIR}/PyTorch2.5_cu2.6.1_Py3.10.sif"
VENV_PATH="${CONTAINER_DIR}/VirtEnv"

CONFIG_DIR="${FINETUNE_DIR}/config_scripts"
PYTHON_DIR="${FINETUNE_DIR}/python_scripts"

# QA
CONFIG_FILE="${CONFIG_DIR}/11B_lora_multi_device.yaml"
PYTHON_FILE="${PYTHON_DIR}/lora_finetune_distributed.py"

# Define the output & logging directories for fine-tuning results
OUTPUT_DIR="$MyWD/data/Llama-3.2-11B-Vision-Instruct_out_multiGPU_srun-tune"
LOGGING_DIR="$MyWD/data/lora_finetune_11B_output_multiGPU_srun-tune"

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

echo "--- My Directory: ${MyWD}"
echo "--- My FineTune Directory: ${FINETUNE_DIR}"
echo "--- My Container Directory: ${CONTAINER_DIR}"
echo "--- My Config-Files Directory: ${CONFIG_DIR}"
echo

# --- Create the Inner Script ---
# Use a temporary file for the inner script to avoid conflicts and ensure atomicity.
INNER_SCRIPT_TEMP="${CONFIG_DIR}/.my_script_temp_${SLURM_JOB_ID}"

# --- Slurm setting
N=$SLURM_JOB_NUM_NODES
nproc_perN=$SLURM_NTASKS_PER_NODE
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "--nbr of nodes: $N"
echo "--nbr of GPUs: $nproc_perN"
echo

# Set up variables to control distributed PyTorch training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

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

# Enables asynchronous error handling for PyTorch
# allows NCCL errors to be reported asynchronously
# and allows other ranks to continue some operations (specific error logging from individual ranks rather than a hard crash)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Flash Attention for efficiency
# Flash Attention is a highly optimized algorithm for computing the "attention mechanism," which is a core component of the Transformer architecture.
export USE_FLASH_ATTENTION=1

# Make all CUDA kernel launches synchronous
export CUDA_LAUNCH_BLOCKING=1

# Set up variables to control distributed PyTorch training
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

echo "Running fine-tuning command:"

# Example of overriding output (comment if NOT needed)
#python "${PYTHON_FILE}" --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1

# Default usage (no overrides)
#python "${PYTHON_FILE}" --config "${CONFIG_FILE}"


# Syntax of "tune run" command
#the flag --standalone is Useful when launching single-node, multi-worker job
#If --standalone specified then the options --rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values are ignored.

tune run --nnodes $N --nproc_per_node $nproc_perN --standalone lora_finetune_distributed --config "${CONFIG_FILE}" checkpointer.output_dir="${OUTPUT_DIR}" output_dir="${LOGGING_DIR}" epochs=1
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
