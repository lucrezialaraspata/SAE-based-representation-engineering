import torch
import torch.nn as nn
import os
from pathlib import Path
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from torch.utils.hooks import RemovableHandle
from functools import partial
import torch.nn as nn
import traceback
from spare.function_extraction_modellings.function_extraction_llama import LlamaForCausalLM
from datasets import load_dataset



PROJ_DIR = Path(os.environ.get("PROJ_DIR", "./"))
MODEL_PATH = "/home/lucrezia/SAE-based-representation-engineering/checkpoints_save_latest/Meta-Llama-3-8B/nqswap/prob_conflict/hidden/prob_model_list_16_L1factor3.pt"


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, use_bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)

    @torch.inference_mode()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x.to(self.linear.weight.device)
        return torch.sigmoid(self.linear(x)).squeeze(1)


class InspectOutputContext:
    def __init__(self, model, module_names, move_to_cpu=False, last_position=False):
        self.model = model
        self.module_names = module_names
        self.move_to_cpu = move_to_cpu
        self.last_position = last_position
        self.handles = []
        self.catcher = dict()

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.module_names:
                handle = inspect_output(module, self.catcher, module_name, move_to_cpu=self.move_to_cpu,
                                        last_position=self.last_position)
                self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            print(f"Value: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False
        return True

def inspect_output(module: nn.Module, catcher: dict, module_name, move_to_cpu, last_position=False) -> RemovableHandle:
    hook_instance = partial(inspect_hook, catcher=catcher, module_name=module_name, move_to_cpu=move_to_cpu,
                            last_position=last_position)
    handle = module.register_forward_hook(hook_instance)
    return handle

def inspect_hook(module: nn.Module, inputs, outputs, catcher: dict, module_name, move_to_cpu, last_position=False):
    if last_position:
        if type(outputs) is tuple:
            catcher[module_name] = outputs[0][:, -1]  # .clone()
        else:
            catcher[module_name] = outputs[:, -1]
        if move_to_cpu:
            catcher[module_name] = catcher[module_name].cpu()
    else:
        if type(outputs) is tuple:
            catcher[module_name] = outputs[0]  # .clone()
        else:
            catcher[module_name] = outputs
        if move_to_cpu:
            catcher[module_name] = catcher[module_name].cpu()
    return outputs



def load_probing_model():
    # First, let's examine what's in the saved file
    saved_data = torch.load(MODEL_PATH, weights_only=True)
    print(f"Type of saved data: {type(saved_data)}")
    print(f"Content: {saved_data}")
    
    model = LogisticRegression(input_dim=4096, use_bias=True)
    
    # Handle different save formats
    if isinstance(saved_data, list):
        # If it's a list, try to extract the state dict from the first element
        if len(saved_data) > 0:
            if hasattr(saved_data[0], 'state_dict'):
                model.load_state_dict(saved_data[0].state_dict())
            elif isinstance(saved_data[0], dict):
                model.load_state_dict(saved_data[0])
            else:
                print(f"Unexpected format in list: {type(saved_data[0])}")
                return
        else:
            print("Empty list found in saved file")
            return
    elif isinstance(saved_data, dict):
        model.load_state_dict(saved_data)
    else:
        print(f"Unexpected save format: {type(saved_data)}")
        return
    
    model.eval()
    print("Model loaded successfully. Ready for inference.")
    return model

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def load_model(model_path, flash_attn, not_return_model=False):
    bnb_config = create_bnb_config()
    n_gpus = torch.cuda.device_count()
    max_memory = "15000MB"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    attn_implementation = "flash_attention_2" if flash_attn else "eager"
    print(f"attn_implementation = {attn_implementation}")
    if not_return_model:
        model = None
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_memory = {i: max_memory for i in range(n_gpus)},
            quantization_config=bnb_config,
        )
        #model.cuda().eval()
        model.eval()
    return model, tokenizer

@torch.no_grad()
def save_activations(
        target_layers=None,
        model_path="meta-llama/Meta-Llama-3-8B",
        #none_conflict=False,
        data_name="demo",
):
    flash_attn = False
    activation_type = "test"

    results_dir = PROJ_DIR / f"cache_data"
    model_name = model_path.split("/")[-1]

    hidden_save_dir = results_dir / model_name / data_name / "activation_hidden" / activation_type
    mlp_save_dir = results_dir / model_name / data_name / "activation_mlp" / activation_type
    attn_save_dir = results_dir / model_name / data_name / "activation_attn" / activation_type
    for sd in [hidden_save_dir, mlp_save_dir, attn_save_dir]:
        if not os.path.exists(sd):
            os.makedirs(sd)

    if target_layers is None:
        if "gemma" in model_name:
            target_layers = list(range(42))
        else:
            target_layers = list(range(32))

    module_names = []
    module_names += [f'model.layers.{idx}' for idx in target_layers]
    module_names += [f'model.layers.{idx}.self_attn' for idx in target_layers]
    module_names += [f'model.layers.{idx}.mlp' for idx in target_layers]

    model, tokenizer = load_model(model_path, flash_attn=flash_attn)

    # dict_messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "A company had 110 employees at the start of Q1 and 100 employees at the end of Q1. How many employees left during Q1?"},
    # ]
    # messages = tokenizer.apply_chat_template(dict_messages, tokenize=False)
    instance_id = 0
    messages = """Question: A company had 110 employees at the start of Q1 and 100 employees at the end of Q1. How many employees left during Q1?
    
    Answer:"""

    tokens = tokenizer(messages, return_tensors="pt")

    with InspectOutputContext(model, module_names) as inspect:
        model(input_ids=tokens["input_ids"].to("cuda"), use_cache=False, return_dict=True)

    for module, ac in inspect.catcher.items():
        # ac: [batch_size, sequence_length, hidden_dim]
        ac_last = ac[0, -1].float()
        layer_idx = int(module.split(".")[2])
        save_name = f"layer{layer_idx}-id{instance_id}.pt"
        if "mlp" in module:
            torch.save(ac_last, mlp_save_dir / save_name)
        elif "self_attn" in module:
            torch.save(ac_last, attn_save_dir / save_name)
        else:
            torch.save(ac_last, hidden_save_dir / save_name)

    combine_activations(model_name, data_name, activation_type=activation_type, layer_ids=target_layers)

def combine_activations(model_name, data_name, analyse_activation=None, activation_type=None, layer_ids=None):
    if layer_ids is None:
        layer_ids = list(range(32))
    if analyse_activation is None:
        analyse_activation = ["mlp", "attn", "hidden"]
    else:
        analyse_activation = [analyse_activation]
    if activation_type is None:
        activation_type = ["conflict", "none_conflict"]  # , "close_book"
    else:
        activation_type = [activation_type]
    results_dir = PROJ_DIR / f"cache_data"

    print(f"Model name: {model_name}")
    print(f"Data name: {data_name} (should be 'demo')")
    print(f"Activation types: {activation_type} (should be 'test')")
    print(f"Analyse activation: {analyse_activation} (should be 'mlp', 'attn', 'hidden')")
    print(f"Layer IDs: {layer_ids}")

    for at in activation_type:  # test
        for aa in analyse_activation:   #hidden, mlp, attn
            act_dir = results_dir / model_name / data_name / f"activation_{aa}" / at
            print(f"\nProcessing activations in {act_dir} for {model_name} {data_name} {at} {aa}")

            act_files = list(os.listdir(act_dir))
            print(f"\nFound {len(act_files)} activation files in {act_dir}")
            print(f"\t {act_files}")

            act_files = [f for f in act_files if len(f.split("-")) == 2]
            print(f"\nFiltered to {len(act_files)} valid activation files")
            print(f"\t {act_files}")

            act_files_layer_idx_instance_idx = [
                [act_f, parse_layer_id_and_instance_id(os.path.basename(act_f))]
                for act_f in act_files
            ]
            print(f"\nExtracted layer and instance IDs from activation files: {len(act_files_layer_idx_instance_idx)}\n\t{act_files_layer_idx_instance_idx}")

            # For each layer id (as key), the value contains a list of [activation file, instance id]
            layer_group_files = {lid: [] for lid in layer_ids}
            for act_f, (layer_id, instance_id) in act_files_layer_idx_instance_idx:
                layer_group_files[layer_id].append([act_f, instance_id])
            print(f"\nGrouped activation files by layer ID: {layer_group_files}")
                        
            for layer_id in layer_ids:
                # Sort the files for each layer by instance ID
                layer_group_files[layer_id] = sorted(layer_group_files[layer_id], key=lambda x: x[1])
                
                acts = []
                loaded_paths = []
                for idx, (act_f, instance_id) in enumerate(layer_group_files[layer_id]):
                    assert idx == instance_id
                    acts.append(torch.load(act_dir / act_f))
                    loaded_paths.append(act_dir / act_f)
                acts = torch.stack(acts)
                print(f"{data_name} {model_name} {at} {aa} layer{layer_id} shape: {acts.shape}")
                save_path = act_dir / f"layer{layer_id}_activations.pt"
                torch.save(acts, save_path)
                for p in loaded_paths:
                    os.remove(p)

def parse_layer_id_and_instance_id(s):
    instance_idx = -1
    layer_idx = -1

    try:
        layer_s, id_s = s.split("-")
        layer_idx = int(layer_s[len("layer"):])
        instance_idx = int(id_s[len("id"):-len(".pt")])
    except Exception as e:
        print(s)
    return layer_idx, instance_idx


def load_activations(
        model_name="meta-llama/Meta-Llama-3-8B",
        data_name="demo",
        analyse_activation="hidden",
        activation_type="test",
        layer_idx=None,
) -> torch.Tensor:
    results_dir = PROJ_DIR / f"cache_data"
    model_name = model_name.split("/")[-1]
    act_dir = results_dir / model_name / data_name / f"activation_{analyse_activation}" / activation_type
    return torch.load(act_dir / f"layer{layer_idx}_activations.pt", map_location="cuda")



@torch.no_grad()
def logistic_regression_eval(model, hidden_state):
    model.eval()

    #output = model(hidden_state.cuda())
    output = model(hidden_state)
    predicted = (output > 0.5).float()
    print(f"Prediction: {predicted}")

def main():
    print("\n" + "="*60)
    print("EXAMPLE USAGE: Knowledge Conflict Classification")
    print("="*60)

    print("\n1. Save a test activation")
    save_activations(target_layers=list(range(15, 26)))
    
    
    print("\n2. Load probing model")
    model = load_probing_model()
    model.to("cuda")
    
    print("\n3. Load activation")
    activation = load_activations(layer_idx=16)

    logistic_regression_eval(model, activation)
    

    print("-" * 60)




if __name__ == "__main__":
    main()