# LLM Workshop Guide: Fine-Tuning & Inference with LLaMA on HPC

## Table of Contents
- [1. Workshop Overview](#1-workshop-overview)
- [2. Environment Setup](#2-environment-setup)
  - [2.1. Connect to Olivia HPC](#21-connect-to-olivia-hpc)
  - [2.2. Run Setup Script](#22-run-setup-script)
- [3. Building & Testing the Container](#3-building--testing-the-container-container)
  - [3.1. Build Custom PyTorch Container](#31-build-custom-pytorch-container)
  - [3.2. Test the Container](#32-test-the-container)
- [4. Fine-Tuning Workflows](#4-fine-tuning-workflows)
  - [4.1. Single GPU Fine-Tuning](#41-single-gpu-fine-tuning-fine-tuning-singlegpu)
  - [4.2. Multi-GPU Fine-Tuning](#42-multi-gpu-fine-tuning-fine-tuning-multigpu)
- [5. Inference](#5-inference-inference)
- [6. Monitoring & Visualization](#6-monitoring--visualization-tools)
- [7. Profiling](#7-profiling-profiling)
- [8. Exercises](#8-exercises-exercise)
- [9. Hands-On Workflow Summary](#9-hands-on-workflow-summary)
- [10. Workshop Exploration Ideas](#10-workshop-exploration-ideas)

---

## 1. Workshop Overview
This workshop provides hands-on training for **fine-tuning and deploying LLaMA-based Large Language Models (LLMs)** for **summarization (XSum)** and **question answering (QA)**.  

It is designed for **HPC environments** with support for **single-GPU and multi-GPU** setups using **SLURM job scheduling**.

### What You’ll Learn
- ✅ Fine-tune **LLaMA models** (LoRA & full fine-tuning)  
- ✅ Run **inference** to generate answers & summaries  
- ✅ Use **single-GPU and multi-GPU** configurations  
- ✅ Execute workloads in **HPC clusters** with SLURM  
- ✅ **Monitor & profile GPU usage**  

> **Note**: All datasets are pre-cleaned and stored in `data/`. No preprocessing needed.  

---

## 2. Environment Setup

### 2.1. Connect to Olivia HPC
```bash
ssh your-username@olivia.sigma2.no
mkdir /cluster/work/projects/nn9997k/$USER
cd /cluster/work/projects/nn9997k/$USER
git clone https://github.com/HichamAgueny/llm-workshop.git
cd llm-workshop
````

### 2.2. Run Setup Script

```bash
chmod u+x my_script.sh
./my_script.sh
```

This will:

* Copy the **Singularity container** and datasets to your project workspace.
* Update configuration file paths automatically.

---

## 3. Building & Testing the Container (`container/`)

### 3.1. Build Custom PyTorch Container

1. SSH to the build node:

   ```bash
   ssh x1000c2s0b0n0
   ```
2. Set temp dirs for Apptainer:

   ```bash
   export APPTAINER_TMPDIR=/tmp/$USER
   export APPTAINER_CACHEDIR=/tmp/$USER
   ```
3. Pull NVIDIA PyTorch base container:

   ```bash
   cd /cluster/work/projects/nn9970k/$USER/llm-workshop/container
   apptainer pull pytorch2.5_cu2.6.1_py3.10.sif \
       docker://nvcr.io/nvidia/pytorch@sha256:618162...
   ```
4. Build custom container:

   ```bash
   apptainer build pytorch2.5_cu2.6.1_py3.10_custom.sif pytorch2.5_cu2.6.1_py3.10_arm.def
   ```

### 3.2. Test the Container

Start an interactive job:

```bash
salloc -A NN9997K -t 00:15:00 -p accel -N 1 --gpus 1 --mem-per-cpu 8G
ssh <allocated-node>
cd container/
apptainer shell --nv pytorch2.5_cu2.6.1_py3.10_custom.sif
```

Inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__config__.show())"
```

---

## 4. Fine-Tuning Workflows

### 4.1. Single GPU Fine-Tuning (`fine-tuning-singlegpu/`)

1. Navigate to job scripts:

   ```bash
   cd fine-tuning-singlegpu/jobs
   vi job_singleGPU_QA.sh
   ```
2. Check config:

   ```bash
   cd ../config_scripts
   vi 1B_lora_single_device_QA.yaml
   ```

   * Adjust LoRA rank, dropout, learning rate, etc.
3. Submit job:

   ```bash
   sbatch job_singleGPU_QA.sh
   ```
4. Monitor GPU usage:

   ```bash
   squeue --me
   monitor_singleGPU <JobID>
   ```
5. Results:

   * Logs: `data/lora_finetune_output_onlyLoRAweight/`
   * LoRA weights: `data/Llama-3.2-1B-Instruct_out_onlyLoRAweight/`

---

### 4.2. Multi-GPU Fine-Tuning (`fine-tuning-multigpu/`)

1. Navigate to job scripts:

   ```bash
   cd fine-tuning-multigpu/jobs
   vi job_multiGPU_QA.sh
   ```
2. Check config:

   ```bash
   cd ../config_scripts
   vi 1B_lora_multi_device_QA.yaml
   ```
3. Submit job:

   ```bash
   sbatch job_multiGPU_QA.sh
   ```
4. Monitor usage:

   ```bash
   monitor_multiGPU <JobID>
   ```
5. Results:

   * Logs: `data/lora_finetune_QA_output_4GPU/`
   * LoRA weights: `data/Llama-3.2-1B-Instruct_QA_out_4GPU_onlyLoRAweight/`

---

## 5. Inference (`inference/`)

1. Navigate to jobs:

   ```bash
   cd inference/jobs
   vi job_inference_QA.sh
   ```
2. Check prompts:

   ```bash
   cd ../../data/prompts
   vi prompt_QA.txt
   ```
3. Submit job:

   ```bash
   sbatch job_inference_QA.sh
   ```
4. Outputs are stored in `inference/jobs/out/`.

---

## 6. Monitoring & Visualization (`tools/`)

### Plot Training Metrics

1. Start interactive session:

   ```bash
   salloc -A NN9997K -t 00:15:00 -p accel -N 1 --gpus 1 --mem-per-cpu 8G
   ssh <allocated-node>
   ```
2. Load modules & install:

   ```bash
   module load cray-python/3.11.7
   pip install matplotlib
   ```
3. Run plotter:

   ```bash
   python tools/plot_training_metrics.py
   ```

This plots:

* **tokens/sec per GPU** vs training steps
* **Peak memory (active & reserved)** vs training steps

---

## 7. Profiling (`profiling/`)

### 7.1. Enable Profiling

Edit config file:

```yaml
profiler:
  enabled: True
  output_dir: /cluster/.../data/profiling_outputs
  cpu: True
  cuda: True
  profile_memory: True
  wait_steps: 5
  warmup_steps: 3
  active_steps: 2
```

### 7.2. Run Profiling Job

```bash
cd profiling/jobs
sbatch job_singleGPU_profiling.sh
```

### 7.3. Transfer Results

```bash
scp -r -J USER@betzy USER@olivia:/cluster/.../profiling_outputs/iteration_10 .
```

### 7.4. View Results

* Open HTML reports locally:

  ```bash
  open rank0_memory-timeline.html
  ```
* Or launch TensorBoard:

  ```bash
  pip install tensorboard
  tensorboard --logdir=iteration_10
  ```

---

## 8. Exercises (`exercise/`)

### 8.1. Summarization (XSum)

* Fine-tune with **LoRA** vs **full fine-tuning**.
* Compare **single vs multi-GPU scaling**.
* Adjust hyperparameters (LR, batch size, epochs).

Scripts & configs are under:

* `exercise/fine-tuning-singlegpu/`
* `exercise/fine-tuning-multigpu/`

### 8.2. Inference (Summarization)

1. Edit prompt:

   ```bash
   vi data/prompts/prompt_Xsum.txt
   ```
2. Submit job:

   ```bash
   cd exercise/inference/jobs
   sbatch job_inference_Xsum.sh
   ```

---

## 9. Hands-On Workflow Summary

1. Explore **SLURM job scripts**.
2. Review **LoRA configs**.
3. Submit jobs & monitor GPUs.
4. Inspect logs & metrics.
5. Tune hyperparameters.
6. Run inference for QA & summarization.
7. Visualize training metrics.
8. Profile system efficiency.

---

## 10. Workshop Exploration Ideas

* Compare **LoRA vs full fine-tuning**.
* Benchmark **single vs multi-GPU** scaling.
* Experiment with **dropout, rank, LR**.
* Profile **memory bottlenecks**.
