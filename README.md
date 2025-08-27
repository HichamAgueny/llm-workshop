## 🧠 Workshop Overview
This repository contains hands-on materials for fine-tuning and deploying **LLaMA-based Large Language Models (LLMs)** for **summarization** and **question answering (QA)**. 
The workshop is designed for execution on **HPC (High-Performance Computing) systems**, with support for both single and multi-GPU configurations on a single node.


This workshop includes:

- ✅ Fine-tuning **LLaMA models** for summarization (e.g. XSum) and QA
- ✅ Running inference to generate summaries and answers
- ✅ Utilizing **single-GPU or multi-GPU setups** setup
- ✅ Executing everything on **HPC environments** with cluster tools (e.g., SLURM)
- ✅ Monitoring the GPU usage

> 📝 Note: All datasets are assumed to be clean and stored in the `data/` directory. No pre-processing required.

## Guide to running the workflows in this repository:

## SSH to Olivia

First, connect to **Olivia** via SSH. Then run the following commands:

```bash
mkdir /cluster/work/projects/nn9997k/$USER
cd /cluster/work/projects/nn9997k/$USER
git clone https://github.com/HichamAgueny/llm-workshop.git
cd llm-workshop
````

### Setup Script

Run the setup script:

```bash
chmod u+x my_script.sh
./my_script.sh
```

This script will:

* Copy the Singularity image and dataset to your project work area.
* Update paths in the configuration files automatically.

## Running Slurm-Based Jobs

This guide explains how to run fine-tuning and inference jobs (slurm) on an HPC system.

---

## Fine-Tuning

### Single GPU
```bash
cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-singlegpu/jobs
ls
sbatch job_singleGPU_QA.sh
````

* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-singlegpu/jobs/out
  ```

#### Monitoring GPU Usage

```bash
cp /cluster/work/projects/nn9997k/$USER/llm-workshop/tools/monitor_singleGPU.sh $HOME/.local/bin/monitor_singleGPU
monitor_singleGPU <JobID>
```

You can find the `<JobID>` by running:

```bash
squeue --me
```

---

### Multiple GPUs

```bash
cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-multigpu/jobs
ls
sbatch job_multiGPU_QA.sh
```

* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-multigpu/jobs/out
  ```

#### Monitoring GPU Usage

```bash
cp /cluster/work/projects/nn9997k/$USER/llm-workshop/tools/monitor_multiGPU.sh $HOME/.local/bin/monitor_multiGPU
monitor_multiGPU <JobID>
```

You can find the `<JobID>` by running:

```bash
squeue --me
```

---

## Inference

### QA Task

```bash
cd /cluster/work/projects/nn9997k/$USER/llm-workshop/inference/jobs
ls
sbatch job_inference_QA.sh
```

* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9997k/$USER/llm-workshop/inference/jobs/out
  ```

---

## Exercise: Summarization Task

Follow the same steps as above, replacing the QA script with the summarization job script.

Fine-Tuning exercise is located in ```/cluster/work/projects/nn9997k/$USER/llm-workshop/exercise```

Inference is located in ```/cluster/work/projects/nn9997k/$USER/llm-workshop/inference```

---

## Workshop Program – Fine-Tuning LLMs with Multi-GPU Training on Olivia

## 09:30 – 10:15 | Introduction to Fine-Tuning & Optimization Strategies (30min + 15min)
- Overview of LLM and Fine-tuning
- Concepts: Parameter-efficient methods (LoRA, LoRA Dropout)
- Model merging

**10:15 – 10:30** | Break (15 min)

## 10:30 – 11:30 | Environment Setup & Configuration Overview (Olivia)
- Overview of Olivia Supercomputer
- Setting up training environment - **interactive session**
- Config file deep dive of Llama  
*(60 min — slightly extended for Q&A)*

## 11:30 – 12:30 | Lunch Break

---

## Hands-On Session – Practical Fine-Tuning & Inference

### Task 1: Question-Answering (Alpaca dataset)

**12:30 – 12:45** | Walk-through of exercise structure  
**12:45 – 13:30** | Fine-tuning on 1 GPU + Inference (45 min)   

**13:30 – 13:45** | Break (15 min)  

**13:45 – 14:00** | Profiling (15 min)

**14:00 – 14:10** | Walk-through for multi-GPU run  
**14:10 – 14:45** | Fine-tuning on multiple GPUs + Inference (35 min)   

**14:45 – 15:00** | Break (15 min)  

---

### Task 2: Choose your own experiment — open-ended experiment
**15:00 – 15:15** | Wrap-up & Discussion (for early leavers)

**15:15 – 16:00** | Hands-on continuity

    - Try e.g. summarization (Xsum dataset) with LoRA vs. full fine-tuning.
    - Compare single vs. multi-GPU scaling.
    - Adjust config parameters (learning rate, batch size, epochs) and compare outputs.
    
## 📁 Repository Structure
llm-workshop/

├── container/ # Environment & singularity container

├── data/ # Clean, ready-to-use datasets for summarization and QA

├── download_xsum.txt # Optional script to download pre-cleaned datasets

├── exercise/ # Guided notebooks and exercises

├── fine-tuning-multigpu/ # Multi-GPU fine-tuning Example

├── fine-tuning-singlegpu/ # Single-GPU fine-tuning Example

├── inference/ # Scripts for inference (summarization and QA)

├── install.sh # Setup script for HPC environments

├── profiling/ # profiling training on a single GPU

├── tools/ # Utility functions for GPU monitoring and Jobs

├── Test/ # Simple GPU test (e.g. CUDA availability)

└── README.md # Project overview and instructions

## 📜 License

This repository contains code under multiple licenses:

- **Original workshop code**: [MIT License](LICENSE)
- **Third-party code from Meta Platforms, Inc.**: BSD-style license. See headers in individual files and Meta’s LICENSE file.

Please ensure compliance with all applicable licenses when using or modifying this repository.
