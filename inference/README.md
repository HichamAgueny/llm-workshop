# Inference - QA task

## Overview
This folder contains script files for inference with QA task. 

## Step-by-Step Workflow

1. **Explore the SLURM job script**  
   - Navigate to the `jobs/` directory:
     ```bash
     cd /cluster/work/projects/nn9997k/$USER/llm-workshop/inference
     ````
   - Open and read `vi job_inference_QA.sh` to understand how the job is structured.

2. **Check the Prompts folder**  
   - Go to the `prompts/` directory:
     ```bash
      cd /cluster/work/projects/nn9997k/hich/llm-workshop/data/prompts
     ````
   - Open the relevant file (`vi prompt_QA.txt`) and experiment by writing different prompts.  

3. **Submit your job**  
   ```bash
   cd /cluster/work/projects/nn9997k/hich/llm-workshop/inference/jobs
   sbatch job_inference_QA.sh
   ````
* The slurm output file is saved in:

  ```
  /cluster/work/projects/nn9997k/hich/llm-workshop/inference/jobs/out
  ```
