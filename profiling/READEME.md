# Profiling

This folder contains tools and instructions for profiling GPU usage and memory consumption during training, but also kernels. Profiling helps you analyze how efficiently your model uses system resources.

---

## 1. Enabling Profiling in Config

To generate profiling data, you first need to enable and configure the profiler in the config file located at:
```bash
/cluster/work/projects/nn9997k/$USER/llm-workshop/profiling/config_scripts
````


A minimal example extracted from ```1B_lora_single_device_QA_profiling.yaml```:

```yaml
# Profiler (enabled)
profiler:
  _component_: torchtune.training.setup_torch_profiler
  enabled: True

  # Output directory for trace artifacts
  output_dir: /cluster/projects/nn9997k/$USER/llm-workshop/data/profiling_outputs

  # Activities to trace
  cpu: True
  cuda: True

  # Trace options
  profile_memory: True
  with_stack: False
  record_shapes: True
  with_flops: True

  # Scheduling options
  wait_steps: 5
  warmup_steps: 3
  active_steps: 2
  num_cycles: 1
````
**Key Points**

**enabled: True** → turns on profiling.

**output_dir** → where profiling results are stored.

**cpu / cuda** → what to trace.

**wait_steps, warmup_steps, active_steps** → control how many steps are used for warmup and profiling.

For quick experiments, set training to run for ~10 steps only.

## 2. Submitting a Profiling Job

Once profiling is enabled in the config, submit the job script from:
```bash
cd /cluster/work/projects/nn9997k/$USER/llm-workshop/profiling/jobs
sbatch job_singleGPU_profiling.sh
````
This will generate profiling outputs in the directory:
```bash
/cluster/projects/nn9997k/$USER/llm-workshop/data/profiling_outputs/
````

## 3. Copy Profiling Data from Cluster to Local Machine

After the job completes, copy the profiling results from the cluster to your local machine (for analyzing profiling results):

```bash
scp -r -J USERNAME@betzy.sigma2.no USERNAME@olivia.sigma2.no:/cluster/work/projects/nn9997k/USERNAME/llm-workshop/data/profiling_outputs/iteration_10 .
````
-r → copies the folder recursively.

-J → specifies a jump host (Betzy → Olivia).

Specify your **USERNAME** and replace paths as needed if your profiling output is in a different directory.

This will copy the folder **iteration_10** into your current local directory.

## Viewing Profiling Results

The profiling outputs are stored as HTML files (e.g., ```rank0_memory-timeline.html```) which you can view in a web browser.

Use the appropriate command for your system:

**Linux:**
```bash
xdg-open rank0_memory-timeline.html
````

**macOS:**
```bash
open rank0_memory-timeline.html
````

**Windows (CMD):**
```bash
start rank0_memory-timeline.html
````
## Notes

Keep runs short (10 steps) to avoid large files.

Multi-GPU runs will generate multiple reports (e.g., rank0_..., rank1_...).

Clean up old profiling runs to save storage space.

## 5. Viewing Profiling Data in TensorBoard (Optional)

You can also explore profiling results with ```TensorBoard```:

First install ```tensorboard```in your local machine:
```bash
pip install tensorboard
```
and then navigate to where the folder **iteration_10** is stored in your local machine. from there run the command:
```bash
tensorboard --logdir=iteration_10
````
After running the command, you’ll see output indicating that TensorBoard is running, e.g. **TensorBoard 2.20.0 at http://localhost:6006/**

Then open in your browser:
```bash
http://localhost:6006
````

This provides a web-based interface to inspect traces, operator breakdowns, and memory usage etc.
