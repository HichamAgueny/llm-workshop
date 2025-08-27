import time
import argparse # Import the argparse module
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# --- 1. Path Definitions ---
# Use argparse to make paths configurable via command-line arguments
parser = argparse.ArgumentParser(description="Run LoRA inference with a base model and adapter.")
parser.add_argument(
    "--base_model_path",
    type=str,
    required=True,
    help="Path to the directory containing your base Large Language Model (e.g., Llama-3.2-1B-Instruct)."
)
parser.add_argument(
    "--lora_adapter_path",
    type=str,
    required=True,
    help="Path to the directory containing your LoRA adapter weights (e.g., Llama-3.2-1B-Instruct_out_onlyLoRAweight)."
)

args = parser.parse_args()

base_model_path = args.base_model_path
lora_adapter_path = args.lora_adapter_path

# --- 2. Load Tokenizer ---
# Loads the tokenizer from the base model's directory. The tokenizer is essential
# for converting text into numerical input IDs and vice-versa.
# 'use_fast=False' is often used for certain tokenizer implementations or for broader compatibility.
# AutoTokenizer often expects the main configuration (e.g., config.json) at this level
# to determine the tokenizer type, even if tokenizer.model is in a subdirectory.
print(f"Loading tokenizer from: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

# --- 3. Load Base Model ---
# Loads the pre-trained base language model.
# Using 'device_map="cuda"' to ensure all model parts are on a single GPU (cuda:0 by default),
# preventing potential 'Expected all tensors to be on the same device' errors.
# 'torch_dtype=torch.bfloat16' uses bfloat16 precision for memory efficiency and faster computation,
# assuming your hardware supports it.
# 'attn_implementation="eager"' is kept to address potential 'enable_gqa' TypeError.
print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda", # Changed from "auto" to "cuda" for single GPU stability
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)

# --- 4. Load LoRA Adapter on Top of the Base Model (Without Merging Initially) ---
# The PeftModel.from_pretrained() method attaches the LoRA adapter to the base model.
# By default, it does NOT merge the LoRA weights directly into the base model's parameters.
# Instead, the LoRA adjustments are applied dynamically during the forward pass.
print(f"Loading LoRA adapter from: {lora_adapter_path}")
model = PeftModel.from_pretrained(model, lora_adapter_path)


# --- 5. Measure and Print Time to Merge LoRA Weights ---
# This section explicitly merges the LoRA adapter weights into the base model.
# This operation modifies the base model's weights in-place.
print("Starting LoRA weight merge operation...")
start_time = time.time()
model = model.merge_and_unload() # This performs the merge
end_time = time.time()
merge_duration = end_time - start_time
print(f"LoRA weights merged to base model in {merge_duration:.4f} seconds.")


# --- 6. Model Preparation for Inference ---
# Moves the merged model to the CUDA device (GPU).
# This line is kept for clarity, though 'device_map="cuda"' already places it there.
print("Moving model to GPU...")
model.cuda()
# Sets the model to evaluation mode. This disables features like dropout,
# which are only relevant during training, ensuring consistent inference results.
model.eval()
print("Model loaded and set to evaluation mode.")

# --- 7. Example Prompt ---
# The text prompt that the model will use to generate a response.
prompt = "--Summarize this text: High-Performance Computing (HPC) refers to the use of supercomputers and parallel processing techniques for solving complex computational problems at high speeds. These systems perform quadrillions of calculations per second, far surpassing the capabilities of standard desktop or server machines. HPC plays a critical role in both academic research and industrial applications, driving advancements in fields such as climate modeling, aerospace engineering, genomics, and financial modeling.
    At its core, HPC relies on the integration of thousands—or even millions—of processing cores working in parallel to perform highly demanding tasks. These systems typically consist of tightly coupled clusters of nodes, high-speed interconnects, and parallel file systems to manage enormous volumes of data efficiently. Software used in HPC is often optimized for scalability, enabling it to take full advantage of the available hardware resources.
    One of the most impactful uses of HPC is in scientific research. For instance, researchers use HPC to simulate the behavior of molecules at the atomic level, enabling breakthroughs in drug discovery and materials science. In climate science, HPC is used to model global weather patterns with high accuracy, which is essential for understanding climate change and developing mitigation strategies. In engineering, simulations of airflow over aircraft wings or stress analysis in bridges are performed using HPC to reduce the need for physical prototypes.
    Industries are also harnessing HPC for real-time data analytics, artificial intelligence (AI), and machine learning. These tasks require processing massive datasets, training complex models, and delivering insights quickly—something HPC systems are uniquely suited to handle. For example, the automotive industry uses HPC to develop autonomous driving algorithms and simulate crash tests in virtual environments.
    As demands for computational power continue to rise, the future of HPC is increasingly tied to innovations like quantum computing, energy-efficient processor architectures, and hybrid cloud infrastructure. Moreover, initiatives like exascale computing aim to push performance to new levels, enabling systems to execute a billion billion (10^18) operations per second.
.\n\n--Output:"
print(f"\nPrompt: {prompt}")

# --- 8. Prepare Inputs for the Model ---
# Tokenizes the prompt and converts it into PyTorch tensors.
# These tensors are then moved to the same device as the model (GPU).
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# --- 9. Generate Text ---
# 'with torch.no_grad():' disables gradient calculations, which is crucial for inference
# as it saves memory and speeds up the process.
# 'model.generate()' performs the text generation.
# - 'max_new_tokens': The maximum number of tokens to generate in the output.
# - 'temperature': Controls the randomness of the generation (lower = less random).
# - 'top_k': Samples from the top_k most likely next tokens.
print("Generating text...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.6,
        top_k=300
    )

# --- 10. Decode and Print Output ---
# Decodes the generated numerical 'outputs' back into human-readable text.
# FIX: Sliced the outputs tensor to only decode the newly generated tokens,
# by starting from the length of the input_ids.
generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("\nGenerated Output:")
print(generated_text)
