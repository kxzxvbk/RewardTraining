import warnings
import os

import torch
from utils import load_pair_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from peft import PeftModel

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    print(f"Script args: \n {script_args}")
    print(f"Training args: \n {training_args}")
    print(f"Model config: \n {model_config}")
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map='auto',
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    model = PeftModel.from_pretrained(model, training_args.output_dir).base_model.model
    sd = torch.load(os.path.join(training_args.output_dir, 'final.pt'))
    model.load_state_dict(sd)
    model = model.eval().cuda()

    ##############
    # Load dataset
    ##############
    dataset = load_pair_dataset(script_args.dataset_name)['train']

    for i in range(len(dataset["train"])):
        sample_chosen = dataset[i]['chosen']
        sample_rejected = dataset[i]['rejected']

        sample_chosen = tokenizer.encode(sample_chosen, add_special_tokens=True, return_tensors='pt').cuda()
        sample_rejected = tokenizer.encode(sample_rejected, add_special_tokens=True, return_tensors='pt').cuda()

        r1 = model.forward(sample_chosen).logits.item()
        r2 = model.forward(sample_rejected).logits.item()

        print(f"Result, accepted: {r1}, rejected: {r2}, margin: {r1 - r2}")
