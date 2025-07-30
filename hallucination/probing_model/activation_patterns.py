from spare.analysis.activation_patterns import activation_analysis, draw_features


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET = "nqswap"
LAYERS = list(range(32))
USE_LOCAL = True


def main():
    print("--"*50)
    print("Analysis of Activation Patterns")
    print("--"*50)

    print("\nConfiguration:")
    print(f"\t- Model: {MODEL_NAME}")
    print(f"\t- Dataset: {DATASET}")
    print(f"\t- Target Layers: {LAYERS}")

    print("\n1. Conflict instances")
    activation_analysis(
        target_layers=LAYERS,
        model_path=MODEL_NAME,
        none_conflict=False,
        data_name=DATASET,
        use_local=USE_LOCAL,
    )

    print("\n2. None-Conflict instances")
    activation_analysis(
        target_layers=LAYERS,
        model_path=MODEL_NAME,
        none_conflict=False,
        data_name=DATASET,
        use_local=USE_LOCAL,
    )

    print("\n3. Draw Features")
    draw_features()
    print("--"*50)


if __name__ == "__main__":
    main()
