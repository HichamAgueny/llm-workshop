# Overview - Fine-tuning for summarization (Xsum dataset)
This folder contains scripts and configuration files for fine-tuning LLMs on a single GPU and multiple GPUs using LoRA, as well as inference.

In this exercise: 
- Try summarization (Xsum dataset) with LoRA vs. full fine-tuning.
- Compare single vs. multi-GPU scaling.
- Adjust config parameters (learning rate, batch size, epochs) and compare outputs.
      
# Single GPU Fine-Tuning

## Step-by-Step Workflow

1. **Explore the SLURM job script**  
   - Navigate to the `jobs/` directory:
     ```bash
     cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-singlegpu/jobs
     ````
   - Open and read `vi job_singleGPU_Xsum.sh` to understand how the job is structured.

2. **Check the configuration file**  
   - Go to the `config/` directory:
     ```bash
      cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-singlegpu/config_scripts
     ````
   - Open the relevant config file (e.g., `vi 1B_lora_single_device_Xsum.yaml`) and experiment with different parameters for LoRA.  

3. **Submit your job**  
   ```bash
   cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-singlegpu/jobs
   sbatch job_singleGPU_Xsum.sh
   ````
* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-singlegpu/jobs/out
  ```

4. **Monitoring GPU Usage**

```bash
export PATH="/cluster/work/projects/nn9970k/$USER/llm-workshop/tools/bin:$PATH"
monitor_singleGPU <JobID>
```

You can find the `<JobID>` by running:

```bash
squeue --me
```

5. **Inspect and visualize results**

Look at the output logs for training metrics:
```bash
   cd /cluster/work/projects/nn9970k/$USER/llm-workshop/data/lora_finetune_Xsum_output
   ls
   ````
Optionally, plot metrics to visualize performance.

LoRA weights are saved in:
```bash
   /cluster/work/projects/nn9970k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct_Xsum_out_OnlyLoRAweight
````
and the base model is saved in ```/cluster/work/projects/nn9970k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct```

---

# Multiple GPUs Fine-Tuning

## Step-by-Step Workflow

1. **Explore the SLURM job script**  
   - Navigate to the `jobs/` directory:
     ```bash
     cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-multigpu/jobs
     ````
   - Open and read `vi job_multiGPU_Xsum.sh` to understand how the job is structured.

2. **Check the configuration file**  
   - Go to the `config/` directory:
     ```bash
      cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-multigpu/config_scripts
     ````
   - Open the relevant config file (e.g., `vi 1B_lora_multi_device_Xsum.yaml`) and experiment with different parameters for LoRA.  

3. **Submit your job**  
   ```bash
   cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-multigpu/jobs
   sbatch job_multiGPU_Xsum.sh
   ````
* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/fine-tuning-multigpu/jobs/out
  ```

4. **Monitoring GPU Usage**

```bash
export PATH="/cluster/work/projects/nn9970k/$USER/llm-workshop/tools/bin:$PATH"
monitor_multiGPU <JobID>
```

You can find the `<JobID>` by running:

```bash
squeue --me
```

5. **Inspect and visualize results**

Look at the output logs for training metrics:
```bash
   cd /cluster/work/projects/nn9970k/$USER/llm-workshop/data/lora_finetune_Xsum_output_4GPU
   ls
   ````
Optionally, plot metrics to visualize performance.

LoRA weights are saved in:
```bash
   /cluster/work/projects/nn9970k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct_Xsum_out_4GPU_onlyLoRAweight
````
and the base model is saved in ```/cluster/work/projects/nn9970k/$USER/llm-workshop/data/Llama-3.2-1B-Instruct```

---

# Inference - Summarization task

## Step-by-Step Workflow

1. **Explore the SLURM job script**  
   - Navigate to the `jobs/` directory:
     ```bash
     cd /cluster/work/projects/nn9970k/$USER/llm-workshop/exercise/inference
     ````
   - Open and read `vi job_inference_Xsum.sh` to understand how the job is structured.

2. **Check the Prompts folder**  
   - Go to the `prompts/` directory:
     ```bash
      cd /cluster/work/projects/nn9970k/hich/llm-workshop/data/prompts
     ````
   - Open the relevant file (`vi prompt_Xsum.txt`) and experiment by providing different prompts.  

3. **Submit your job**  
   ```bash
   cd /cluster/work/projects/nn9970k/hich/llm-workshop/exercise/inference/jobs
   sbatch job_inference_Xsum.sh
   ````
* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9970k/hich/llm-workshop/exercise/inference/jobs/out
  ```

