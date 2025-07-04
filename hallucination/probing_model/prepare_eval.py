from spare.prepare_eval import main as prepare_eval


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET = "nqswap"
USE_LOCAL = True


def main():
    print("--"*50)
    print("Prepare Activations before Training Probing Model")
    print("--"*50)

    print("\nConfiguration:")
    print(f"\t- Model: {MODEL_NAME}")
    print(f"\t- Dataset: {DATASET}")

    print("\n1. Conflict instances")
    prepare_eval(
        flash_attn=False,
        exp_name=f"{DATASET}-llama3-8b-conflict",
        model_path=MODEL_NAME,
        batch_size=1,
        run_open_book=True,
        run_close_book=False,
        dataset_name=DATASET,
        use_local=USE_LOCAL,
    )

    print("\n2. None-Conflict instances")
    prepare_eval(
        flash_attn=False,
        exp_name=f"{DATASET}-llama3-8b-conflict",
        model_path=MODEL_NAME,
        batch_size=1,
        run_open_book=False,
        run_close_book=True,
        dataset_name=DATASET,
        use_local=USE_LOCAL,
    )

    print("--"*50)


if __name__ == "__main__":
    main()
