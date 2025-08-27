from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# --- 1. Path Definitions ---
# Path to the directory containing your base Large Language Model.
base_model_path = "/cluster/projects/nn9997k/hicham/llm-workshop/data/Llama-3.2-1B-Instruct"
# Path to the directory containing your LoRA adapter weights.
# The 'adapter_model.safetensors' or similar file will be inside this folder.
lora_adapter_path = "/cluster/projects/nn9997k/hicham/llm-workshop/data/XLlama-3.2-1B-Instruct_out_onlyLoRAweight"

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
# FIX: Changed device_map from "auto" to "cuda" to ensure all model parts are on a single GPU,
# preventing the 'Expected all tensors to be on the same device' error.
# 'torch_dtype=torch.bfloat16' uses bfloat16 precision for memory efficiency and faster computation,
# assuming your hardware supports it.
# 'attn_implementation="eager"' is kept to address potential 'enable_gqa' TypeError.
print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda", # Changed from "auto" to "cuda"
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)

# --- 4. Load LoRA Adapter on Top of the Base Model (Without Merging) ---
# The PeftModel.from_pretrained() method attaches the LoRA adapter to the base model.
# By default, it does NOT merge the LoRA weights directly into the base model's parameters.
# Instead, the LoRA adjustments are applied dynamically during the forward pass.
# This keeps the base model untouched and allows for easy swapping of LoRA adapters.
print(f"Loading LoRA adapter from: {lora_adapter_path}")
model = PeftModel.from_pretrained(model, lora_adapter_path)

# --- 5. Model Preparation for Inference ---
# Moves the model (base + LoRA adapter) to the CUDA device (GPU).
# This line might become redundant if device_map="cuda" works, but it's harmless to keep.
print("Moving model to GPU...")
model.cuda()
# Sets the model to evaluation mode. This disables features like dropout,
# which are only relevant during training, ensuring consistent inference results.
model.eval()
print("Model loaded and set to evaluation mode.")

# --- 6. Example Prompt ---
# The text prompt that the model will use to generate a response.
prompt = "--Explain the importance of High Performance Computing in modern research in a concise way.\n\n--Output:"
print(f"\nPrompt: {prompt}")

# --- 7. Prepare Inputs for the Model ---
# Tokenizes the prompt and converts it into PyTorch tensors.
# These tensors are then moved to the same device as the model (GPU).
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# --- 8. Generate Text ---
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

# --- 9. Decode and Print Output ---
# Decodes the generated numerical 'outputs' back into human-readable text.
# 'skip_special_tokens=True' removes any special tokens (e.g., padding tokens) from the output.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Output:")
print(generated_text)

