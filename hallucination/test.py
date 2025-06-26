import torch
import torch.nn as nn

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

MODEL_PATH = "/home/lucrezia/SAE-based-representation-engineering/checkpoints_save_latest/Meta-Llama-3-8B/nqswap/prob_conflict/hidden/prob_model_list_16_L1factor3.pt"


def main():
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


if __name__ == "__main__":
    main()