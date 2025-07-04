from spare.kc_probing import draw_probing_model_accuracy as plot_accuracy


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET = "nqswap"
LAYERS = list(range(10, 25))
ACTIVATION_TYPES = ["hidden", "mlp", "attn"]


def main():
    print("--"*50)
    print("Plotting Probing Model Accuracy")
    print("--"*50)

    plot_accuracy(model_path=MODEL_NAME, data_name=DATASET, l1_factor=3)
    
    print("--"*50)


if __name__ == "__main__":
    main()
