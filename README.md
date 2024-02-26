# llm-mcts

This is a project to explore Monte-Carlo Tree Search (MCTS) for Code-Gen tasks. We first test our method on the Human-Eval dataset, and extend to the Verilog-Eval dataset. For a detailed explanation of the experiments please see the accompanying blog at /web/index.html also hosted at https://localhost:3000/web/index.html. 

## Env Setup

This project uses `conda` to manage its python environment and packages. To install all relevant libraries, run the following:

```sh
conda env create -f environment.yml
conda activate llm-mcts
```

## Human-Eval

We use a modified human-eval dataset/enviroment from https://github.com/arunpatro/human-eval. This fork contains updated code for python-3.10 and also extends the error feedback to include the traceback.

```sh
git clone https://github.com/arunpatro/human-eval
cd human-eval && pip install -e .
```

Checkout the `nbs/humaneval.ipynb` for a demo.

### Run experiments

```sh
python src/baselines.py
python src/mcts.py
```

## Verilog-Eval

We use a modified verilog-eval dataset/enviroment from https://github.com/arunpatro/verilog-eval. This fork contains updated code for python-3.10 and also extends the error feedback to include the traceback, vcdcat for further waveform analysis.

```sh
git clone https://github.com/arunpatro/verilog-eval
cd verilog-eval && pip install -e .
git clone https://github.com/cirosantilli/vcdvcd
cd vcdvcd && pip install -e .
```

### Setup Icarus Verilog

Executing tests from the verilog-eval dataset requires a local installation of [iverilog](https://github.com/steveicarus/iverilog). You'll need to follow the relevant [installation steps](https://github.com/steveicarus/iverilog#buildinginstalling-icarus-verilog-from-source) to get it setup. Once this is done, run the following to verify everything is working correctly:

```sh
iverilog -V
vvp -V
```

### Run experiments

```sh
PYTHONPATH="./verilog" python src/baselines.py verilog
PYTHONPATH="./verilog" python src/mcts.py verilog
```
