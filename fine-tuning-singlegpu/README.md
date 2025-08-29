# Single GPU Fine-Tuning

## Overview
This folder contains scripts and configuration files for fine-tuning LLMs on a single GPU using LoRA.

## Step-by-Step Workflow

1. **Explore the SLURM job script**  
   - Navigate to the `jobs/` directory:
     ```bash
     cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-singlegpu/jobs
     ````
   - Open and read `vi job_singleGPU_QA.sh` to understand how the job is structured.

2. **Check the configuration file**  
   - Go to the `config/` directory:
     ```bash
      cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-singlegpu/config_scripts
     ````
   - Open the relevant config file (e.g., `vi 1B_lora_single_device_QA.yaml`) and experiment with different parameters for LoRA.  

3. **Submit your job**  
   ```bash
   cd /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-singlegpu/jobs
   sbatch job_singleGPU_QA.sh
   ````
* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9997k/$USER/llm-workshop/fine-tuning-singlegpu/jobs/out
  ```

4. **Monitoring GPU Usage**
A script for monitoring the GPU usage and GPU memory utilization is made available in:
```
/cluster/work/projects/nn9997k/$USER/llm-workshop/tools/bin
```
And this path ```export PATH="/cluster/work/projects/nn9997k/$USER/llm-workshop/tools/bin:$PATH"``` is already added to your `.bashrc`. You simply need to source it:
```bash
source ~/.bashrc
monitor_singleGPU <JobID>
```

You can find the `<JobID>` by running:

```bash
squeue --me
```

5. **Inspect and visualize results**

Look at the output logs for training metrics:
```bash
   cd /cluster/work/projects/nn9997k/$USER/llm-workshop/data/lora_finetune_output_onlyLoRAweight
   ls
   ````
Optionally, plot metrics to visualize performance.

LoRA weights are saved in:
```bash
   /cluster/work/projects/nn9997k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct_out_onlyLoRAweight
````
and the base model is saved in ```/cluster/work/projects/nn9997k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct```
