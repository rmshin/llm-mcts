import outlines
from concurrent.futures import ThreadPoolExecutor
from human_eval.execution import check_correctness
from human_eval.data import HUMAN_EVAL, stream_jsonl

problems = list(stream_jsonl(HUMAN_EVAL))
executor = ThreadPoolExecutor(max_workers=10)

# fmt: off
# list of task_ids that the baseline generated samples failed to solve even once
HARD_PROBLEMS = ['HumanEval/103', 'HumanEval/105', 'HumanEval/106', 'HumanEval/107', 'HumanEval/108', 'HumanEval/109', 'HumanEval/111', 'HumanEval/113', 'HumanEval/115', 'HumanEval/119', 'HumanEval/120', 'HumanEval/123', 'HumanEval/124', 'HumanEval/125', 'HumanEval/126', 'HumanEval/127', 'HumanEval/128', 'HumanEval/129', 'HumanEval/130', 'HumanEval/132', 'HumanEval/133', 'HumanEval/134', 'HumanEval/137', 'HumanEval/138', 'HumanEval/140', 'HumanEval/141', 'HumanEval/142', 'HumanEval/143', 'HumanEval/144', 'HumanEval/145', 'HumanEval/146', 'HumanEval/147', 'HumanEval/148', 'HumanEval/151', 'HumanEval/156', 'HumanEval/160', 'HumanEval/163', 'HumanEval/19', 'HumanEval/20', 'HumanEval/26', 'HumanEval/32', 'HumanEval/33', 'HumanEval/38', 'HumanEval/39', 'HumanEval/40', 'HumanEval/46', 'HumanEval/54', 'HumanEval/57', 'HumanEval/59', 'HumanEval/6', 'HumanEval/67', 'HumanEval/68', 'HumanEval/71', 'HumanEval/72', 'HumanEval/75', 'HumanEval/81', 'HumanEval/83', 'HumanEval/85', 'HumanEval/89', 'HumanEval/91', 'HumanEval/93', 'HumanEval/94', 'HumanEval/95', 'HumanEval/96', 'HumanEval/97']
# TEMP: list of task_ids that are to be skipped due to added complexity in measuring pass-rate
# might re-visit these again at a later point if necessary
SKIP_PROBLEM_IDS = ['HumanEval/32', 'HumanEval/38', 'HumanEval/44', 'HumanEval/50', 'HumanEval/53', 'HumanEval/87', 'HumanEval/113', 'HumanEval/151']
# fmt: on
STOP_SEQUENCES = ["```"]


def split_problem_tests(problem):
    pre_base_str, tests = problem["test"].split("def check(candidate):\n")
    base_str = "def check(candidate):\n"
    split_tests = []
    for i in tests.split("\n"):
        if (
            (i.lstrip() == "")
            or (i.lstrip().startswith("#"))
            or (i.lstrip().startswith("print"))
        ):
            continue
        split_tests.append(pre_base_str + base_str + i)
    return split_tests


task_id_problem_map = {problem["task_id"]: problem for problem in problems}
task_id_split_tests_map = {
    problem["task_id"]: split_problem_tests(problem) for problem in problems
}


def stats_execute(task_id, completion, timeout=10):
    problem = task_id_problem_map[task_id]
    split_tests = task_id_split_tests_map[task_id]
    _problem = problem.copy()
    results = []
    for i in split_tests:
        _problem["test"] = i
        future = executor.submit(check_correctness, _problem, completion, timeout)
        result = future.result()
        results.append(result)

    return {
        "task_id": task_id,
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


instructions = "Please answer the following question following the examples. Generate valid python code by indenting 4 spaces always."
examples = problems[:2]


def get_prompts_with_ids():
    prompts_with_ids = [
        (few_shot_prompt(instructions, examples, problem["prompt"]), problem["task_id"])
        for problem in problems[2:]
    ]
    return prompts_with_ids


def get_hard_prompts_with_ids():
    prompts_with_ids = [
        (few_shot_prompt(instructions, examples, problem["prompt"]), problem["task_id"])
        for problem in problems[2:]
        if problem["task_id"] in HARD_PROBLEMS
        and problem["task_id"] not in SKIP_PROBLEM_IDS
    ]
    return prompts_with_ids
