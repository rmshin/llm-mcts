import outlines
from concurrent.futures import ThreadPoolExecutor
from human_eval.execution import check_correctness
from human_eval.data import HUMAN_EVAL, stream_jsonl, write_jsonl

problems = list(stream_jsonl(HUMAN_EVAL))
executor = ThreadPoolExecutor(max_workers=5)

## TODO: should be returning fraction of test cases passed, instead of 0/1
def execute(problem, completion, timeout):
    future = executor.submit(check_correctness, problem, completion, timeout)
    result = future.result()
    return result


def stats_execute(problem, completion, timeout):
    pre_base_str, tests = problem['test'].split('def check(candidate):\n')
    base_str = "def check(candidate):\n"
    split_tests = [pre_base_str + base_str + i for i in tests.split('\n') if i != '']
    
    _problem = problem.copy()
    results = []
    for i in split_tests:
        _problem['test'] = i
        future = executor.submit(check_correctness, _problem, completion, timeout)
        result = future.result()
        results.append(result)
    
    
    return {'task_id': problem['task_id'], 'pass_rate': sum([i['passed'] for i in results])/len(results)}

@outlines.prompt
def few_shots(instructions, examples, question):
    """{{ instructions }}

    {% for example in examples %}
    
    Question:
    ```
    {{ example.prompt }}
    ```
    Answer:
    ```
    {{ example.canonical_solution }}
    ```
    {% endfor %}

    Question:
    ```
    {{ question }}
    ```
    Answer:
    ```
    """

instructions = "Please answer the following question following the examples. Generate valid python code by indenting 4 spaces always."
examples = problems[:2]

from llama_cpp import Llama
from tqdm import tqdm

model = Llama(
    model_path="../deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=256,
    n_threads=8,
    logits_all=True,
)

N_SAMPLES = 20
samples = []
for problem in tqdm(problems[2:20]):
    for _ in tqdm(range(N_SAMPLES)):
        prompt = few_shots(instructions, examples, problem['prompt'])
        output = model(prompt=prompt, max_tokens=80, temperature=0.2, stop=['```'])
        res = output['choices'][0]['text']
        item = dict(task_id=problem['task_id'], completion=res)
        samples.append(item)

write_jsonl("few_shot_baselines.jsonl", samples)

# run this to get the pass@k metrics
# evaluate_functional_correctness samples.jsonl