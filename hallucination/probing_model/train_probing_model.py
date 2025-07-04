from spare.kc_probing import main as train
from tqdm import tqdm


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET = "nqswap"
LAYERS = list(range(10, 25))
ACTIVATION_TYPES = ["hidden", "mlp", "attn"]


def main():
    print("--"*50)
    print("Training Probing Model")
    print("--"*50)

    print("\nConfiguration:")
    print(f"\t- LLM: {MODEL_NAME}")
    print(f"\t- Dataset: {DATASET}")
    print(f"\t- Target Layers: {LAYERS}")
    print(f"\t- Activation Types: {ACTIVATION_TYPES}")

    for act_type in ACTIVATION_TYPES:
        print(f"\n - Activation{act_type}")
        for layer_idx in tqdm(LAYERS, desc="\t\tTraining Layers"):
            train(
                model_name=MODEL_NAME,
                dataset_name=DATASET,
                layer_idx=layer_idx,
                activation_type=act_type,
            )
    
    print("--"*50)


if __name__ == "__main__":
    main()
