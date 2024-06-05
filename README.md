# 🥷ninja-merger
*English* | [日本語](README_ja.md)


ninja-merger is a minimal tool for LLM vector merge.


## Installation

1. Clone repository
   ```bash
   git clone https://github.com/Local-novel-llm-project/ninja-merger
   cd ninja-merger
   ```

1. (Optional but recommended) Create and activate your Python environment
   ```bash
   # for example, we use venv
   python -m venv venv
   ```

1. Install dependencies using pip
   ```bash
   pip install -r requirements.txt
   ```


## Usage

```bash
python ninja-merger.py -c <your yaml config>.yml
```

## Configs

ninja-merger uses the YAML format for configure merge method.
Examples of configuration files can be found in the `examples` folder.

Details of the settings are written in the comments of example configuration file.


## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)