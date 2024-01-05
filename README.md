## llm-mcts
This is a project to explore Monte-Carlo Tree Search for Code-Gen tasks. We first test our method on the Human-Eval dataset. 

### Setup Human-Eval dataset and enviroment
We use a modified human-eval dataset/enviroment from https://github.com/arunpatro/human-eval. This fork contains updated code for python-3.10 and also extends the error feedback to include the traceback.
```sh
pip install -e human-eval
```
Checkout the `nbs/humaneval.ipynb` for a demo.