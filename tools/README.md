## Overview

As Olivia is still in the pilot phase, there are no modules available in the login nodes. Therefore, you need to 
launch an interactive session in order to access available modules:

```bash
salloc -A NN9997K -t 006:30:00 -p accel -N 1 --gpus 1 --mem-per-cpu 8G
```
The output looks like this:
```bash
salloc: Pending job allocation 90341
salloc: job 90341 queued and waiting for resources
salloc: job 90341 has been allocated resources
salloc: Granted job allocation 90341
salloc: Nodes **x1000c0s0b1n0** are ready for job
```
and displays the hostname.
```
ssh x1000c0s0b1n0
cd /cluster/work/projects/nn9970k/$USER/llm-workshop/tools
```

## Plot Training Metrics

A python code `plot_training_metrics.py` is made availble  here `/cluster/work/projects/nn9970k/$USER/llm-workshop/tools`. The code:
1. Reads the logs from `/cluster/work/projects/nn9997k/hich/llm-workshop/data/lora_finetune_output_onlyLoRAweight/log_1756467542.txt`
2. Plots `tokens_per_second_per_gpu` as a function Training steps.
3. Plots `peak_memory_active` and `peak_memory_reserved` as a function of Training steps'

To run the python code:
1. Load this module `module load cray-python/3.11.7`
2. Set proxy settings for HTTP and HTTPS traffic:
   
`export http_proxy=http://10.63.2.48:3128/`

`export https_proxy=http://10.63.2.48:3128/`

4. Install matplotlib `pip install matplotlib`
5. run `python plot_training_metrics.py`
