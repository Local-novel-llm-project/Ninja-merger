
# 🥷 Ninja Merger

[English](README.md) | [日本語](README_ja.md)

## Overview

Ninja Merger is a tool for merging PyTorch-based deep learning models, particularly Transformer models.  It allows you to combine multiple models to create new models or fine-tune existing ones. It supports a variety of merging methods (addition, subtraction, mixing, QEIC, etc.), providing flexibility in how you merge your models.

## Features

*   **Diverse Merging Methods:**
    *   Basic arithmetic operations (addition, subtraction, multiplication, division)
    *   Model mixing (Mix, Average)
    *   Tensor concatenation (Concatenation)
    *   Max/Min pooling (MaxPool, MinPool)
    *   Geometric mean (GeometricMean)
    *   Subtraction with standard deviation consideration (StdSub)
    *   Model widening (WidenMerge)
    *   Complex number-based merging (ComplexAdd, ComplexAngleMerge)
    *   Quantum-entanglement-inspired calculation-based merging (QEICAdd, QeicMix, QeicSub)
*   **Flexible Configuration:**
    *   Uses YAML configuration files for detailed control over the merging process.
    *   Manages multiple model merging configurations within a single file.
    *   Allows defining model-specific settings (key transformations, layer insertion).
    *   Enables setting per-layer velocities.
    *   Specifies layers to be merged by range or name.
    *   Offers options for handling layers with mismatched sizes (skip or use only the common part).
*   **LoRA Support:**
    *   Automatically merges LoRA (Low-Rank Adaptation) models.
*   **Detailed Logging:**
    *   Provides visually appealing and informative log output using the `rich` library.
*   **Extensibility:**
    *   Modular structure makes it easy to add new merging methods and pre/post-processing options.

## Installation

```bash
git clone https://github.com/Local-novel-llm-project/Ninja-merger.git
cd ninja-merger
pip install -r requirements.txt
```

## Usage

1.  **Create a Configuration File:** Create a YAML file named `config.yaml` and specify the models to be merged, the merging method, and other options.
2.  **Run the Command:** Execute the following command to merge the models.

```bash
python main.py -c config.yaml -o merged_models
```

## Supported Formats

*   `*.safetensors`
*   AutomodelForCausalLM (HuggingFace)

## Configuration File Example

```yaml
models:
- name: "model_add"
  left: "path/to/model1"
  right: "path/to/model2"
  operation: "add"
- name: "model_mix_recurrent"
  left: "model_add"  # result of the previous merge operation
  right: "path/to/model3"
  operation: "mix"
  velocity: 0.5
```

## Command-Line Arguments

```
usage: main.py [-h] [-c CONFIG] [-o OUT_DIR] [-n] [-dm MERGE_MODELS_DEVICE] [-dt TARGET_MODEL_DEVICE] [-t TORCH_DTYPE] [-r RECURRENT_MODE] [-d] [-l]
               [--dump_layers] [--include_layers INCLUDE_LAYERS] [--exclude_layers EXCLUDE_LAYERS]

Merge models

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to the JSON configuration file
  -o OUT_DIR, --out_dir OUT_DIR
                        Directory to save the merged model
  -n, --skip_layernorm  Skip layernorm during merging
  -dm MERGE_MODELS_DEVICE, --merge_models_device MERGE_MODELS_DEVICE
                        Device for merging models
  -dt TARGET_MODEL_DEVICE, --target_model_device TARGET_MODEL_DEVICE
                        Device for the target model
  -t TORCH_DTYPE, --torch_dtype TORCH_DTYPE
                        Torch data type
  -r RECURRENT_MODE, --recurrent_mode RECURRENT_MODE
                        use target recurrent mode
  -d, --dry_run         Dump processed layer infos without merging
  -l, --save_only_last_model
                        Only last model saved
  --dump_layers         Dump model layers to a file instead of merging
  --include_layers INCLUDE_LAYERS
                        Comma-separated list of layers to include
  --exclude_layers EXCLUDE_LAYERS
                        Comma-separated list of layers to exclude
```

## Contributing

Bug reports, feature requests, and pull requests are welcome.

## License

Apache License 2.0

