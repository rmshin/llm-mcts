# llm-mcts

This is a project to explore Monte-Carlo Tree Search (MCTS) for Code-Gen tasks. We first test our method on the Human-Eval dataset.

## Setup

This project uses `conda` to manage its python environment and packages. To install all relevant libraries, run the following:

```sh
conda env create -f environment.yml
conda activate llm-mcts
```

### Setup Human-Eval

We use a modified human-eval dataset/enviroment from https://github.com/arunpatro/human-eval. This fork contains updated code for python-3.10 and also extends the error feedback to include the traceback.

```sh
git clone https://github.com/arunpatro/human-eval
cd human-eval && pip install -e .
```

Checkout the `nbs/humaneval.ipynb` for a demo.

### Running MCTS code generation

```sh
PYTHONPATH="./human-eval" python src/mcts.py
```

### Setup Verilog-Eval

We use a modified verilog-eval dataset/enviroment from https://github.com/arunpatro/verilog-eval. This fork contains updated code for python-3.10 and also extends the error feedback to include the traceback, vcdcat for further waveform analysis.

```sh
git clone https://github.com/arunpatro/verilog-eval
cd verilog-eval && pip install -e .
git clone https://github.com/cirosantilli/vcdvcd
cd vcdvcd && pip install -e .
```
