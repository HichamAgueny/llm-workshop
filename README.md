## Table of Contents
- [Workshop Overview](#workshop-overview)
- [Setup Instructions](#setup-instructions)
- [Folder Descriptions](#folder-descriptions)
- [Hands-On Workflow](#hands-on-workflow)
- [Workshop Program](#workshop-program)

## Workshop Overview
This repository contains hands-on materials for fine-tuning and deploying **LLaMA-based Large Language Models (LLMs)** for **summarization** and **question answering (QA)**. 
The workshop is designed for execution on **HPC (High-Performance Computing) systems**, with support for both single and multi-GPU configurations on a single node.


This workshop includes:

- ✅ Fine-tuning **LLaMA models** for QA and summarization (e.g. XSum)
- ✅ Running inference to generate answers and summaries
- ✅ Utilizing **single-GPU or multi-GPU setups** setup
- ✅ Executing everything on **HPC environments** with cluster tools (e.g., SLURM)
- ✅ Monitoring the GPU usage

> 📝 Note: All datasets are assumed to be clean and stored in the `data/` directory. No pre-processing required.

## Setup Instructions

### SSH to Olivia

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

## Folder Descriptions
- `container/` – Singularity container for the HPC environment.
- `data/` – Pre-cleaned datasets for QA and summarization tasks.
- `exercise/` – Guided notebooks and hands-on exercises.
- `fine-tuning-singlegpu/` – Single GPU fine-tuning example.
- `fine-tuning-multigpu/` – Multi-GPU fine-tuning example.
- `inference/` – Scripts for running inference.
- `profiling/` – GPU profiling.
- `tools/` – Utility scripts for monitoring and job management.
- `test/` – Small scripts to test GPU availability and environment setup.
  
## Hands-On Workflow
1. Explore SLURM job scripts.
2. Review configuration files for LoRA FT.
3. Submit jobs and monitor GPU usage.
4. Inspect logs, metrics, and visualize results.
5. Experiment with hyperparameters (e.g. LoRA rank, dropout).
6. Perform inference on QA & summarization tasks.

Detailed instructions for each step are provided in the corresponding folder README files:
 
 - Instructions for Fine-tuning on a single GPU: [fine-tuning-singlegpu/README.md](fine-tuning-singlegpu/README.md)
 - Instructions for Fine-tuning on multiple GPUs: [fine-tuning-multigpu/README.md](fine-tuning-multigpu/README.md) 
 - Instructions for inference: [inference/README.md](inference/README.md)
 - Guided exercises for practice are described here: [exercise/README.md](exercise/README.md)

## Workshop Program 

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
    

## 📜 License

This repository contains code under multiple licenses:

- **Original workshop code**: [MIT License](LICENSE)
- **Third-party code from Meta Platforms, Inc.**: BSD-style license. See headers in individual files and Meta’s LICENSE file.

Please ensure compliance with all applicable licenses when using or modifying this repository.
