import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import gc
import os
import json

# Load YAML configuration
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Use fast weight downloads/uploads
HF_HUB_ENABLE_HF_TRANSFER=True

model_slug = config.get('model_slug')
model_name = model_slug.split('/')[-1]
tokenizer_slug = config.get('tokenizer_slug', None)
if tokenizer_slug is None:
    tokenizer_slug = model_slug
use_4bit = config.get('4bit', False)
flash_attention_2 = config.get('flash_attention_2', True)
dtype = config.get('dtype', 'bfloat16')
chat_template_path = config.get('chat_template', None)

# Print descriptive message
print(f"\n\nLoading model and tokenizer with the following settings:")
print(f"- Model: {model_slug}")
print(f"- Tokenizer: {tokenizer_slug}")
print(f"- 4-bit Quantization: {'Enabled' if use_4bit else 'Disabled'}")
print(f"- Flash Attention 2: {'Enabled' if flash_attention_2 else 'Disabled'}")
print(f"- Data Type: {dtype}")
if chat_template_path:
    print(f"- Chat template: {chat_template_path}")

# Set the output directory
output_dir = 'test_output'
os.makedirs(output_dir, exist_ok=True)

# Define BitsAndBytesConfig if 4bit quantization is enabled
quantization_config = None
if use_4bit:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, dtype)
    )
    quantization_config = bnb_config

# Load model and tokenizer
from_pretrained_kwargs = {
    "quantization_config": quantization_config,
    "torch_dtype": getattr(torch, dtype),
    "trust_remote_code": True,
    "device_map": 'auto',
    "cache_dir": '',
}

# Conditionally add 'flash_attention_2' if it's not None
if flash_attention_2 is not None and flash_attention_2:
    from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"

# Create the model with dynamically constructed arguments
model = AutoModelForCausalLM.from_pretrained(
    model_slug,
    **from_pretrained_kwargs
)

# # Print the model configuration
# print(f"Model config:\n{model.config}\n")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_slug, trust_remote_code=True)

# Load a custom chat template
if chat_template_path:
    with open(chat_template_path, 'r') as file:
        chat_template = file.read()
    tokenizer.chat_template = chat_template.replace('    ','').replace('\n','')

print(f"Model: \n{model}")
print(f"Model Config: \n{model.config}")
print(f"Model: \n{tokenizer}")