## 🧠 Workshop Overview
This repository contains hands-on materials for fine-tuning and deploying **LLaMA-based Large Language Models (LLMs)** for **summarization** and **question answering (QA)**. 
The workshop is designed for execution on **HPC (High-Performance Computing) systems**, with support for both single and multi-GPU configurations on a single node.


This workshop includes:

- ✅ Fine-tuning **LLaMA models** for summarization (e.g. XSum) and QA (e.g. SQuAD)
- ✅ Running inference to generate summaries and answers
- ✅ Utilizing **single-GPU or multi-GPU setups** setup
- ✅ Executing everything on **HPC environments** with cluster tools (e.g., SLURM)
- ✅ Monitoring the GPU usage

> 📝 Note: All datasets are assumed to be clean and stored in the `data/` directory. No pre-processing required.

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

├── tools/ # Utility functions for GPU monitoring and Jobs

├── Test/ # Simple GPU test (e.g. CUDA availability)

└── README.md # Project overview and instructions
