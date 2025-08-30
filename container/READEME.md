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
