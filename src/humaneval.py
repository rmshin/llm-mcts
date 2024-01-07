import outlines
from concurrent.futures import ThreadPoolExecutor
from human_eval.execution import check_correctness
from human_eval.data import HUMAN_EVAL, stream_jsonl

problems = list(stream_jsonl(HUMAN_EVAL))
executor = ThreadPoolExecutor(max_workers=10)

STOP_SEQUENCES = ["```"]
num_few_shot_examples = 2
examples = problems[:num_few_shot_examples]
instructions = "Please answer the following question following the examples. Generate valid python code by indenting 4 spaces always."


def stats_execute(prompt_idx, completion, timeout=10):
    problem = problems[prompt_idx + num_few_shot_examples]
    pre_base_str, tests = problem["test"].split("def check(candidate):\n")
    base_str = "def check(candidate):\n"
    split_tests = []
    for i in tests.split("\n"):
        if (i.lstrip() == "") or (i.lstrip().startswith("#")):
            continue
        split_tests.append(pre_base_str + base_str + i)

    _problem = problem.copy()
    results = []
    for i in split_tests:
        _problem["test"] = i
        future = executor.submit(check_correctness, _problem, completion, timeout)
        result = future.result()
        results.append(result)

    return {
        "task_id": problem["task_id"],
        "pass_rate": sum([i["passed"] for i in results]) / len(results),
    }


@outlines.prompt
def few_shot_prompt(instructions, examples, question):
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


def get_prompts():
    prompts = [
        few_shot_prompt(instructions, examples, problem["prompt"])
        for problem in problems[2:]
    ]
    return prompts
