import argparse
import os
from safetensors.torch import load_file as load_safetensors
import torch

def extract_keys(model_path):
    """
    Extracts keys from a .safetensors or .pth model file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith(".safetensors"):
        state_dict = load_safetensors(model_path)
        return list(state_dict.keys())
    elif model_path.endswith(".pth"):
        checkpoint = torch.load(model_path, map_location="cpu")

        # Common keys for state_dict
        potential_keys = ["state_dict", "weight", "model", "net_g"]

        for key in potential_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return list(checkpoint[key].keys())

        # If no common key is found, return the top-level keys
        if isinstance(checkpoint, dict):
            return list(checkpoint.keys())
        else:
            raise ValueError("Could not find a state dictionary in the .pth file.")

    else:
        raise ValueError("Unsupported file format. Please use .safetensors or .pth")

def save_keys_to_file(keys, model_path):
    """
    Saves the extracted keys to a text file.
    """
    output_path = os.path.splitext(model_path)[0] + ".txt"
    with open(output_path, "w") as f:
        for key in keys:
            f.write(f"{key}\n")
    print(f"Layer info extracted to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract layer information from a model file.")
    parser.add_argument("model_path", type=str, help="Path to the .safetensors or .pth model file.")
    args = parser.parse_args()

    try:
        keys = extract_keys(args.model_path)
        save_keys_to_file(keys, args.model_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
