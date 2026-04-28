# Ninja Merger

[English](README.md) | [日本語](README_ja.md)

Ninja Merger merges PyTorch and Hugging Face model files from YAML
recipes. It can save a full model when you provide an explicit `target`, save a
sparse vector artifact when `target` is `null`, chain steps through `recurrent`,
filter or drop layers, and apply registered merge operations such as arithmetic,
angle-based, Widen, and QEIC merges.

This README is the quick entry point. Detailed, code-aligned references live in
[docs/README.md](docs/README.md); those detailed pages are currently maintained
primarily in Japanese.

## Current Capabilities

- YAML-driven multi-step merge recipes under the required `models` section.
- Model inputs from Hugging Face model ids/directories, `*.safetensors`, `*.pth`,
  `*.bin`, and Ninja Merger `*.difftensors` artifacts.
- Full target-model output when `target` points to an existing model.
- Sparse vector output as `*.difftensors` when `target` is `null` / `none`.
- Previous-step reuse through `recurrent` in `left`, `right`, or `target`.
- Layer inclusion, exclusion, dropping, layer-name dumping, and mismatch handling.
- Per-layer `velocity` / `post_velocity` values with prefix or regex matching.
- Operation registry covering basic arithmetic, `passthrough`, `none`, Widen,
  angle/complex merges, and QEIC operations. `complex_mix` is registered but not
  implemented.
- Rich progress display, per-step summaries, and saved recipe files.

## Installation

Python `>=3.13` is declared in `pyproject.toml`.

```bash
git clone https://github.com/Local-novel-llm-project/Ninja-merger.git
cd Ninja-merger
uv sync
```

If you are not using `uv`, install the runtime dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

Use an explicit `target` when you want Ninja Merger to save a full model output.
The default `post_operation` is `add`, so set `post_operation: none` when the
merge result itself should replace the target layer value.

```yaml
models:
  - name: mix-full-model
    target: path/to/target-model
    left: path/to/base-model
    right: path/to/tuned-model
    operation: mix
    velocity: 0.35
    post_operation: none
```

Run the merge:

```bash
python main.py -c config.yaml -o merged_models --merge-models-device cuda:0 --target-model-device cuda:0 --torch-dtype bfloat16
```

To create a sparse vector artifact instead of a full model, use `target: null`.
Ninja Merger uses the first `left` model as the in-memory layer source and saves
only non-zero tensors to `merged_models/vector/*.difftensors`; it does not save a
tokenizer or full model directory. Because `post_operation` still defaults to
`add`, set `post_operation: none` for a plain delta such as `left - right`.

```yaml
models:
  - name: delta-vector
    target: null
    left: path/to/base-model
    right: path/to/tuned-model
    operation: sub
    post_operation: none
```

## CLI Essentials

```bash
python main.py -c model_config.yaml -o merged_models
```

Common options:

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML merge recipe. Default: `model_config.yaml`. |
| `-o`, `--out-dir` | Output directory. Default: `./merged_models`. |
| `-dm`, `--merge-models-device` | Device for `left` / `right` models. Default: `cpu`. |
| `-dt`, `--target-model-device` | Device for explicit `target` models. Default: `cpu`. |
| `-t`, `--torch-dtype` | One of `float16`, `bfloat16`, `float32`, `float64`. |
| `--no-recurrent-mode` | Disable previous-step reuse. |
| `-l`, `--save-only-last-model` | Keep intermediate steps in memory and save only the final step. |
| `-d`, `--dry-run` | Run the merge path without saving output artifacts. |
| `--dump-layers` | Write layer names instead of merging. Cannot be combined with `--dry-run`. |
| `--include-layers`, `--exclude-layers` | CLI layer filters. They override per-entry filters. |

See [docs/cli.md](docs/cli.md) for the complete CLI reference.

## Documentation

- [docs/README.md](docs/README.md): documentation index. Detailed reference
  pages are currently maintained primarily in Japanese.
- [docs/configuration.md](docs/configuration.md): YAML schema, defaults, examples.
- [docs/operations.md](docs/operations.md): merge operations and modifiers.
- [docs/output-artifacts.md](docs/output-artifacts.md): inputs, outputs, and artifact layout.
- [docs/cli.md](docs/cli.md): command-line behavior.
- [docs/examples-and-tools.md](docs/examples-and-tools.md): examples and auxiliary tools.

## Development

```bash
pytest
```

The tests exercise parser behavior, config normalization, operation registry
validation, layer handling, sparse vector artifacts, and generated reference
cases for implemented merge operations.

## License

Apache License 2.0
