import os
import json
import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from spare.sae import Sae
from spare.function_extraction_modellings.function_extraction_gemma2 import Gemma2ForCausalLM
from spare.function_extraction_modellings.function_extraction_llama import LlamaForCausalLM
from spare.sae_lens.eleuther_sae_wrapper import EleutherSae
from typing import Union, Any


PROJ_DIR = Path(os.environ.get("PROJ_DIR", "./"))
HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")


def add_file_handler(_logger, output_dir: str, file_name: str):
    file_handler = logging.FileHandler(os.path.join(output_dir, file_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s"))
    _logger.addHandler(file_handler)


def load_jsonl(path):
    with open(path, "r") as fn:
        data = [json.loads(line) for line in fn.readlines()]
    return data


def get_weight_dir(
    model_ref: str,
    *,
    model_dir: Union[str, os.PathLike[Any]] = HF_DEFAULT_HOME,
    revision: str = "main",
    repo_type="models"
) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist or is not a directory."

    model_path = Path(os.path.join(model_dir, "hub", "--".join([repo_type, *model_ref.split("/")])))
    assert model_path.is_dir(), f"Model path {model_path} does not exist or is not a directory."
    
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir(), f"Weight directory {weight_dir} does not exist or is not a directory."

    if repo_type == "datasets":
        # For datasets, we need to return the directory containing the dataset files
        weight_dir = weight_dir / "data"
    
    return weight_dir


def load_model(model_path, flash_attn, not_return_model=False, use_local=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    n_gpus = torch.cuda.device_count()
    max_memory = "15000MB"

    if not use_local:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            truncation_side="left",
        )
    else:
        model_local_path = get_weight_dir(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_local_path, local_files_only=True, padding_side='left', truncation_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    attn_implementation = "flash_attention_2" if flash_attn else "eager"
    print(f"attn_implementation = {attn_implementation}")
    if not_return_model:
        model = None
    else:
        if "gemma" in model_path.lower():
            model = Gemma2ForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=bnb_config,
                #trust_remote_code=True,
                #max_memory = {i: max_memory for i in range(n_gpus)},
            )
        else:
            if not use_local:
                model = LlamaForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    max_memory = {i: max_memory for i in range(n_gpus)},
                )
            else:
                model_local_path = get_weight_dir(model_path)
                model = LlamaForCausalLM.from_pretrained(
                    model_local_path,
                    local_files_only=True,
                    attn_implementation=attn_implementation,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    max_memory = {i: max_memory for i in range(n_gpus)},
                )
        model.cuda().eval()
    return model, tokenizer


def init_frozen_language_model(model_path, attn_imp="flash_attention_2"):
    bf16 = torch.bfloat16
    
    if "llama" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation=attn_imp, torch_dtype=bf16)
    elif "gemma" in model_path:
        model = Gemma2ForCausalLM.from_pretrained(model_path, attn_implementation=attn_imp, torch_dtype=bf16)
    else:
        raise NotImplementedError
    
    model.to('cuda')
    model.cuda().eval()
    
    for pn, p in model.named_parameters():
        p.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_frozen_sae(layer_idx, model_name):
    if model_name == "Llama-3.1-8B":
        sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint=f"layers.{layer_idx}")
    elif model_name == "Llama-2-7b-hf":
        sae = Sae.load_from_hub("yuzhaouoe/Llama2-7b-SAE", hookpoint=f"layers.{layer_idx}")
    elif model_name == "gemma-2-9b":
        sae, cfg_dict, sparsity = EleutherSae.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical",
            sae_id=f"layer_{layer_idx}/width_131k/canonical",
            device="cuda"
        )
    else:
        raise NotImplementedError(f"sae for {model_name}")
    for pn, p in sae.named_parameters():
        p.requires_grad = False
    sae.cuda()
    return sae
