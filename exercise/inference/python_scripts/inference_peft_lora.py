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
    help="Path to the directory containing your LoRA adapter weights (e.g., XLlama-3.2-1B-Instruct_out_onlyLoRAweight)."
)
parser.add_argument(
    "--prompt_file",
    type=str,
    required=True,
    help="Path to a text file containing the prompt for text generation."
)

args = parser.parse_args()

base_model_path = args.base_model_path
lora_adapter_path = args.lora_adapter_path
prompt_file_path = args.prompt_file # Get the prompt file path

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

# --- 7. Load Prompt from File ---
# Reads the prompt content from the specified file.
try:
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    print(f"\nLoaded prompt from {prompt_file_path}:\n{prompt}")
except FileNotFoundError:
    print(f"Error: Prompt file not found at {prompt_file_path}")
    exit()
except Exception as e:
    print(f"Error reading prompt file: {e}")
    exit()

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
# Sliced the outputs tensor to only decode the newly generated tokens,
# by starting from the length of the input_ids.
generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("\nGenerated Output:")
print(generated_text)
