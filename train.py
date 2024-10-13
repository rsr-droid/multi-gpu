# Run this with FSDP with "accelerate launch train.py" after having run "accelerate config"
# YOU MUST RUN "accelerate config" before running this script. See the README.md for options to select.
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi
from accelerate import PartialState
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' #can help a little with VRAM reqs.

# Load configuration
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load the model and tokenizer
device_map = config['device_map']
model_slug = config['model_slug']
tokenizer_slug = config.get('tokenizer_slug', None)
if tokenizer_slug is None:
    tokenizer_slug = model_slug
use_4bit = config.get('4bit', False)
flash_attention_2 = config.get('flash_attention_2', False)
dtype = config.get('dtype', 'bfloat16')
chat_template_path = config.get('chat_template', None)
output_dir=config['training_output_dir']

new_model_tag = config.get('new_model_tag',False)

hf_username=config['hf_username']
base_model_name = model_slug.split("/")[-1]
dataset_name = config['dataset'].split("/")[-1]

fine_tuned_slug=f"SFT-{dataset_name}"

new_model_slug = f"{hf_username}/{base_model_name}-{fine_tuned_slug}"

if new_model_tag:
    new_model_slug = new_model_slug + '-' + new_model_tag

print(f"\nUsing a 'new_model_slug' of {new_model_slug}\n")

# Define the local save path for the model (and adapters, if applicable)
local_save_path_model = f"{new_model_slug}-local"
local_save_path_adapters = f"{new_model_slug}-adapters-local"

if config.get('wandb_project_name',None) is not None:
    os.environ["WANDB_PROJECT"] = config.get('wandb_project_name',None)  # name your W&B project


## Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_slug, trust_remote_code=True,
    add_bos_token=False # Covered by the chat template
    )

# Set left padding
tokenizer.padding_side  = config.get('padding_side','left')

if config.get('eos_token',None) is not None:
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(config.get('eos_token'))
    print(f"Setting tokenizer eos_token_id to {tokenizer.eos_token_id}\n")

# # Rarely, it makes sense to set a bos token (and you can set it to the eos_token)
# if tokenizer.bos_token is None and config.get('add_bos_token',False):
#     # # Add a BOS token, setting it to the same as the EOS token
#     # tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
#     # # Ensure the BOS token ID is set to the same as the EOS token ID
#     # tokenizer.bos_token_id = tokenizer.eos_token_id

# Replace the above block with:
if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
    print(f"The tokenizer has a BOS token: {tokenizer.bos_token}")
    print(f"Note that whether a BOS token is added when tokenizing depends on the specific tokenizer's behavior and the chat template used.\n")

# Ensure a pad token is set in the model and tokenizer
if '<pad>' in tokenizer.get_vocab():
    tokenizer.pad_token = '<pad>'
elif '<|pad|>' in tokenizer.get_vocab():
    tokenizer.pad_token = '<|pad|>'
elif '<unk>' in tokenizer.get_vocab():
    tokenizer.pad_token = '<unk>'
else:
    tokenizer.pad_token = tokenizer.eos_token
    
# Load a custom chat template
if chat_template_path:
    with open(chat_template_path, 'r') as file:
        chat_template = file.read()
    tokenizer.chat_template = chat_template.replace('    ','').replace('\n','')

# Load Dataset
splits = ['train', 'validation']
datasets = {}  # Use 'datasets' to avoid confusion with 'dataset' function

for split in splits:
    # Determine the split name and optional row limit
    split_name = config.get(f'dataset_{split}_split')
    if split_name is None:
        continue  # Skip if no split is configured

    max_rows = config.get(f'dataset_{split}_max_rows')
    if max_rows is not None:
        split_name = f"{split_name}[:{max_rows}]"
    
    # Load the dataset for the current split
    if split == 'train' or (split == 'validation' and not config.get('generate_val_split', False)):
        datasets[split] = load_dataset(config['dataset'], split=split_name, revision=config.get('dataset_branch','main'))
    
    if split == 'train' and config.get('generate_val_split', False):
        # Split the training dataset into training and validation datasets
        train_split_percentage = 0.8  # 80% of the data for training
        datasets['train'], datasets['validation'] = datasets[split].train_test_split(test_size=1 - train_split_percentage).values()

    # Create a new 'messages' column from the specified column, if it doesn't already exist
    if 'messages' not in datasets[split].column_names and config.get('dataset_messages_column_name') is not None:
        datasets[split] = datasets[split].add_column('messages', datasets[split][config['dataset_messages_column_name']])

## Model loading
# Note that 4bit is not support if using DeepSpeed on multi-GPUs
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

# set model loading kwargs
from_pretrained_kwargs = {
    "quantization_config": quantization_config,
    "torch_dtype": getattr(torch, dtype),
    "trust_remote_code": True,
    "cache_dir": '',
    "use_cache": not config.get('gradient_checkpointing', True),
    "attn_implementation": "flash_attention_2" if flash_attention_2 else "eager",
}

# Conditionally add 'device_map' to model loading kwargs
if device_map == "auto":
    from_pretrained_kwargs["device_map"] = device_map
elif device_map == "DDP":
    device_string = PartialState().process_index
    # Ensure device_map is correctly set as a dictionary mapping
    device_map = {'': device_string}
    from_pretrained_kwargs["device_map"] = device_map
elif device_map == "FSDP":
    # If device_map is 'FSDP', no action is required according to the user's requirement.
    pass
elif device_map == "DeepSpeed":
    # If device_map is 'DeepSpeed', no action is required according to the user's requirement.
    pass
else:
    # Throw an error if device_map is not one of the expected values
    raise ValueError("Invalid value for 'device_map'. Only 'auto', 'DDP', or 'FSDP' or 'DeepSpeed' are supported.")

# Create the model with dynamically constructed arguments
model = AutoModelForCausalLM.from_pretrained(
    model_slug,
    **from_pretrained_kwargs
)

# Update pad token id in model and its config
model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

if config.get('eos_token',None) is not None:
    model.config.eos_token_id = tokenizer.eos_token_id
    print(f"Setting model eos_token_id to {model.config.eos_token_id}\n")

print(f"Tokenizer eos is {tokenizer.eos_token}.\n")

from peft import LoftQConfig, prepare_model_for_kbit_training

if use_4bit:
    model = prepare_model_for_kbit_training(model)

    loftq_config = LoftQConfig(loftq_bits=4)
    init_lora_weights = "loftq"
else:
    loftq_config = None
    init_lora_weights = False

## Apply LoRA (if use_lora is True in the config)
if config.get('use_lora', False):
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['lora_modules'],
        lora_dropout=0.1,  # Example value, adjust as needed
        bias="none",  # Example setting, adjust as needed
        task_type="CAUSAL_LM",
        modules_to_save=config.get('other_trainable', None),
        # init_lora_weights=init_lora_weights, # not yet working
        # loftq_config = loftq_config, # not yet working
        use_rslora=config.get('use_rslora',False),
    )
else:
    lora_config=None

# Doesn't work like this if the lora adapter is being passed into the trainer....
    # if use_4bit:
    #     from peft import replace_lora_weights_loftq
    #     replace_lora_weights_loftq(model) #provides better weight initialisation

# Training setup
run_name = new_model_slug.split('/')[1] + '-' + f"lr_{config['learning_rate']}_bs_{config['training_batch_size']}"

# Common Training Configurations
training_args_common = {
    "num_train_epochs": config['num_epochs'],
    "per_device_train_batch_size": config['training_batch_size'],
    "per_device_eval_batch_size": config['validation_batch_size'],
    "gradient_accumulation_steps": config['gradient_accumulation_steps'],
    "do_eval": config['do_eval'],
    "bf16": config['bf16'],
    "learning_rate": float(config['learning_rate']),
    "lr_scheduler_type": config['lr_scheduler_type'],
    "save_steps": config.get('save_steps', 0),
    "optim": "adamw_torch",
    "save_strategy": "steps",
    "evaluation_strategy": "steps",
    "logging_dir": os.path.join(output_dir, "logs", run_name),
    "output_dir": output_dir,
    "eval_steps": 0.25,
    "warmup_ratio": 0.1,
    "logging_steps": 1,
    "hub_private_repo": True,
    "gradient_checkpointing": config.get('gradient_checkpointing', False),
    "save_total_limit": 1,
    "report_to": config.get('report_to','none'),
    "run_name": run_name,
}

if 'gradient_checkpointing' in config and config['gradient_checkpointing'] is True:
    training_args_common['gradient_checkpointing_kwargs'] = {"use_reentrant": config.get('use_reentrant', True)}  # Corrected: removed comma to avoid tuple creation

# Conditionally set 'max_steps' if defined in config
if 'max_steps' in config and config['max_steps'] is not None:
    training_args_common['max_steps'] = config['max_steps']

training_args_common['max_seq_length'] = config['max_seq_length']
    
# SFT-specific configurations
training_args = SFTConfig(**training_args_common)

# Conditionally set 'max_steps' if defined in config
if 'completions_only' in config and config['completions_only'] is True:
    from trl import DataCollatorForCompletionOnlyLM
    response_template_with_context = config['response_template']
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    # instruction_template = '<|im_start|>user'
    # response_template = config['response_template']
    # collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
else:
    collator=None

trainer = SFTTrainer(model=model,
                     tokenizer=tokenizer,
                     args=training_args,
                     train_dataset=datasets['train'],
                     eval_dataset=datasets['validation'],
                     data_collator=collator,
                     peft_config=lora_config
                    )

if config.get('use_lora', False):
    # handle PEFT+FSDP case
    # trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

if hasattr(trainer, 'is_fsdp_enabled') and trainer.is_fsdp_enabled:
    trainer.accelerator.print(f"Commencing Training with:\n")
    trainer.accelerator.print(f"Model:\n{model}\n")
    trainer.accelerator.print(f"Tokenizer:\n{tokenizer}\n")
    trainer.accelerator.print(f"Dataset:\n{datasets}\n")
else:
    print(f"Commencing Training with:\n")
    print(f"Model:\n{model}\n")
    print(f"Tokenizer:\n{tokenizer}\n")
    print(f"Dataset:\n{datasets}\n")

# Training
trainer_stats = trainer.train()

# Save model
if hasattr(trainer, 'is_fsdp_enabled') and trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

# Check if LoRA was used and process accordingly
if config.get('use_lora', False):
    trainer.save_model(local_save_path_adapters)
    tokenizer.save_pretrained(local_save_path_adapters)
    print(f"\nModel and tokenizer have been saved to {local_save_path_adapters}\n.")
else:
    # model.save_pretrained(local_save_path_model, token=True)
    trainer.save_model(local_save_path_model)
    tokenizer.save_pretrained(local_save_path_model)
    print(f"\nModel and tokenizer have been saved to {local_save_path_model}\n.")

# Memory stats - code courtesy: https://github.com/unslothai/unsloth
#@title Show final memory and time stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")
