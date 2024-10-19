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
test_file = config.get('test_file', 'data/messages.json') # Default test file path
save_tag=config.get('test_save_tag',None)
use_dataset_to_test = config['use_dataset_to_test']
dataset_slug = config.get('dataset', None)
dataset_test_branch = config.get('dataset_branch', 'main')
dataset_test_split = config.get('dataset_test_split', 'test')
dataset_test_column_name = config.get('dataset_test_column_name', 'messages')
dataset_test_max_rows = config.get('dataset_test_max_rows', 1)
chat_template_path = config.get('chat_template', None)

# Print descriptive message
print(f"\n\nStarting 'test.py' with the following settings:")
print(f"- Model: {model_slug}")
print(f"- Tokenizer: {tokenizer_slug}")
print(f"- 4-bit Quantization: {'Enabled' if use_4bit else 'Disabled'}")
print(f"- Flash Attention 2: {'Enabled' if flash_attention_2 else 'Disabled'}")
print(f"- Data Type: {dtype}")
if chat_template_path:
    print(f"- Chat template: {chat_template_path}")
print(f"- Use Dataset to Test: {'Enabled' if use_dataset_to_test else 'Disabled'}")
if use_dataset_to_test:
    print(f"- Dataset: {dataset_slug}")
else:
    print(f"- Test File: {test_file}")
print("\n")

# Set the output directory
output_dir = 'test_output'
os.makedirs(output_dir, exist_ok=True)

if save_tag is not None:
    save_name = model_slug.split('/')[1] + '-' + save_tag
else:
    save_name = model_slug.split('/')[1]

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

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_slug, trust_remote_code=True,
    add_bos_token=False # Necessary because your chat template should include this logic
    )

# # Rarely, it makes sense to set a bos token (and you can set it to the eos_token)
# if tokenizer.bos_token is None and config.get('add_bos_token',False):
#     # # Add a BOS token, setting it to the same as the EOS token
#     # tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
#     # # Ensure the BOS token ID is set to the same as the EOS token ID
#     # tokenizer.bos_token_id = tokenizer.eos_token_id

if tokenizer.bos_token is not None:
    print(f"Note that 'add_bos_token' is {tokenizer.add_bos_token}, which determines whether a bos token is added when tokenizing (note that often this is best done in the chat_template instead).\n")

# Load a custom chat template
if chat_template_path:
    with open(chat_template_path, 'r') as file:
        chat_template = file.read()
    tokenizer.chat_template = chat_template.replace('    ','').replace('\n','')

def load_data_from_dataset():
    # Load the specified dataset split, default to 'train' if 'test' split is not available
    try:
        if dataset_test_max_rows is not None:
            dataset = load_dataset(dataset_slug, split=f"{dataset_test_split}[:{dataset_test_max_rows}]", revision=dataset_test_branch)
        else:
            dataset = load_dataset(dataset_slug, split=f"{dataset_test_split}", revision=dataset_test_branch)
    except ValueError:
        dataset = load_dataset(dataset_slug, split=f"train[:{dataset_test_max_rows}]", revision=dataset_test_branch)

    # Extract conversation data
    conversations = dataset[dataset_test_column_name]
    # print("conversations: ", conversations)
    return conversations

def run_inference(data):

    output_file_path = os.path.join(output_dir, f"{save_name}.txt")

    print(f"Writing to: {output_file_path}")

    with open(output_file_path, 'w') as output_file:
        for i, item in enumerate(data):
            prompt_messages=item[:-1]
            correct_response=item[-1:]

            formatted_prompt = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)

            output_file.write(f"** Row {i} **\n")
            # output_file.write(f"Formatted Prompt:\n{formatted_prompt}\n") # note that this will not include special tokens
            # output_file.write(f"Correct Response:\n{correct_response[0]['content']}\n\n")

            inputs = tokenizer([formatted_prompt], return_tensors="pt", truncation=True, max_length=2048).to("cuda")

            model.generation_config.top_k=None

            outputs = model.generate(**inputs,
                            max_new_tokens=250,
                            do_sample=False,
                            # repetition_penalty=1.2, #can help with performance of smaller models
                            # pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            temperature=1.0,
                            top_p=1.0,
                           )
            generated_text = tokenizer.decode(
                outputs[0], # will print inputs and outputs
                # outputs[0, len(inputs.input_ids[0]):], # if you only want to print outputs (not inputs)
                skip_special_tokens=False
                )
        
            output_file.write(f"Prompt + Response:\n{generated_text}\n\n")
            output_file.flush()

            # Clear GPU cache and run garbage collection
            torch.cuda.empty_cache()  # Clear GPU cache
            gc.collect()  # Run garbage collection

# Main logic to decide data source
if use_dataset_to_test:
    print(f"Ignoring test_file and loading data from the {dataset_test_branch} branch of {dataset_slug}, with:")
    print(f"- Dataset Split: {dataset_test_split}")
    print(f"- Dataset Column Name: {dataset_test_column_name}")
    print(f"- Dataset Max Rows: {dataset_test_max_rows}")
    print("\n")
    data = load_data_from_dataset()
    run_inference(data)
else:
    print("Loading data from test file...")
    with open(config['test_file'], 'r') as file:
        data = json.load(file)
    
    run_inference(data)

print(f"Testing complete. The output has been written to {os.path.join(output_dir, save_name)}")