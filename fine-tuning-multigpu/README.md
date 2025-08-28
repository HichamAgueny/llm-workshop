# Multiple GPUs Fine-Tuning

## Overview
This folder contains scripts and configuration files for fine-tuning LLMs on multiple GPUs using LoRA.

## Step-by-Step Workflow

1. **Explore the SLURM job script**  
   - Navigate to the `jobs/` directory:
     ```bash
     cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-multigpu/jobs
     ````
   - Open and read `vi job_multiGPU_QA.sh` to understand how the job is structured.

2. **Check the configuration file**  
   - Go to the `config/` directory:
     ```bash
      cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-multigpu/config_scripts
     ````
   - Open the relevant config file (e.g., `vi 1B_lora_multi_device_QA.yaml`) and experiment with different parameters for LoRA.  

3. **Submit your job**  
   ```bash
   cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-multigpu/jobs
   sbatch job_multiGPU_QA.sh
   ````
* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-multigpu/jobs/out
  ```

4. **Monitoring GPU Usage**

```bash
export PATH="/cluster/work/projects/nn9997k/$USER/llm-workshop/tools/bin:$PATH"
monitor_multiGPU <JobID>
```

You can find the `<JobID>` by running:

```bash
squeue --me
```

5. **Inspect and visualize results**

Look at the output logs for training metrics:
```bash
   cd /cluster/work/projects/nn9997k/$USER/llm-workshop/data/lora_finetune_QA_output_4GPU
   ls
   ````
Optionally, plot metrics to visualize performance.

LoRA weights are saved in:
```bash
   /cluster/work/projects/nn9997k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct_QA_out_4GPU_onlyLoRAweight
````
and the base model is saved in ```/cluster/work/projects/nn9997k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct```
