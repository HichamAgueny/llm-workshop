# Building a Custom Singularity Container for PyTorch

This guide explains how to build a custom PyTorch container using **Apptainer (formerly Singularity)** on the dedicated compute node `x1000c2s0b0n0` on Olivia supercomputer.

---

## Steps

### 1. SSH into the Build Node
A dedicated compute node has been temporarily reserved for container building:
```bash
ssh x1000c2s0b0n0
```
### 2. Set Up Temporary Directories

Define environment variables so that Apptainer uses temporary storage under `/tmp/$USER`:
```bash
export APPTAINER_TMPDIR=/tmp/$USER
export APPTAINER_CACHEDIR=/tmp/$USER
```

### 3. Pull the Base PyTorch Container

Download the official NVIDIA PyTorch container (pinned by digest for reproducibility):
```bash
cd /cluster/work/projects/nn9970k/$USER/llm-workshop/container
apptainer pull pytorch2.5_cu2.6.1_py3.10.sif \
    docker://nvcr.io/nvidia/pytorch@sha256:618162fa0745658a9084745a2a08c38697f09801e15c674ec0fd72658346437b
```
### 4. Clean Up Temporary Directory

Remove temporary files to free space:
```bash
rm -rf /tmp/$USER
```

### 5. Add Extra Packages

Update the container with additional packages as defined in the .def file:

**pytorch2.5_cu2.6.1_py3.10_arm.def**

### 6. Build the Custom Container

Use the definition file to build a new image:
```bash
apptainer build --ignore-fakeroot-command \
    pytorch2.5_cu2.6.1_py3.10_custom.sif \
    pytorch2.5_cu2.6.1_py3.10_arm.def
```
**Output**

 - Base image: **pytorch2.5_cu2.6.1_py3.10.sif**

 - Custom image: **pytorch2.5_cu2.6.1_py3.10_custom.sif**

## Testing the Container
Exit the building node and launch an interactive session:
```bash
salloc -A NN9997K -t 00:15:00 -p accel -N 1 --gpus 1 --mem-per-cpu 8G
```
and then ssh to the allocated compute node e.g. `ssh x1000c0s0b1n0`

### 1. Navigate to the Working Directory
```bash
cd /cluster/work/projects/nn9970k/$USER/llm-workshop/container
```
### 2. Start an Interactive Shell

Run the container with GPU support and bind the current directory:
```bash
apptainer shell --nv -B $PWD:$PWD pytorch2.5_cu2.6.1_py3.10_custom.sif
```
### 3. Run Checks Inside the Container
- Check CUDA Availability
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

- Show PyTorch Build Configuration
```bash
python -c "import torch; print(torch.__config__.show())"
```

- Verify NCCL Linkage
```bash
python -c "import torch; print(torch.cuda.nccl.version())"  # Prints NCCL version used
```

- Check installed packges
```bash
pip list

tune ls
```

## Fine-tuning Setup

Once you have successfully launched the **PyTorch container**, you can set up fine-tuning for **LLaMA 3.2-1B-Instruct** using Hugging Face and the provided tuning recipes.

---

### 1. Download the Model
Use `tune` to fetch the pretrained weights from Hugging Face:
```bash
tune download meta-llama/Llama-3.2-1B-Instruct \
  --output-dir /cluster/work/projects/nn9970k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct \
  --ignore-patterns "original/consolidated*" \
  --hf-token your-hugging-face-token
```
**Note** Replace your-hugging-face-token with a valid Hugging Face access token.

### 2. Copy Configuration Files

Copy the built-in configuration files, respectively, for single GPU, multiple GPUs and inference, into your working directory:
```bash
tune cp llama3_2/1B_lora_single_device .
tune cp llama3_2/1B_lora_multi_device .
tune cp generation .
```

Update the dataset path inside the configuration:
```bash
sed -i "s|/tmp/Llama-3.2-1B-Instruct|/cluster/work/projects/nn9970k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct|g" 1B_lora_single_device.yaml
```

### 3. Copy Fine-Tuning Recipe Scripts

Copy in the fine-tuning (for single GPU & multiple GPUs) and generation python scripts:
```bash
tune cp lora_finetune_single_device .
tune cp lora_finetune_distributed .
tune cp generate .
```
