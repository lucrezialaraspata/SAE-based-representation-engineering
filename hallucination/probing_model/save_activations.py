from spare.analysis.analysis_save_activations import save_activations


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET = "nqswap"
LAYERS = list(range(32))
USE_LOCAL = True


def main():
    print("--"*50)
    print("Save Activations")
    print("--"*50)

    print("\nConfiguration:")
    print(f"\t- Model: {MODEL_NAME}")
    print(f"\t- Dataset: {DATASET}")
    print(f"\t- Target Layers: {LAYERS}")

    print("\n1. Conflict instances")
    save_activations(
        model_path=MODEL_NAME,
        none_conflict=False,
        data_name=DATASET,
        target_layers=LAYERS,
        use_local=USE_LOCAL,
    )

    print("\n2. None-Conflict instances")
    save_activations(
        model_path=MODEL_NAME,
        none_conflict=True,
        data_name=DATASET,
        target_layers=LAYERS,
        use_local=USE_LOCAL,
    )
    print("--"*50)


if __name__ == "__main__":
    main()
