import outlines
from concurrent.futures import ThreadPoolExecutor
from human_eval.execution import check_correctness
from human_eval.data import HUMAN_EVAL, stream_jsonl

problems = list(stream_jsonl(HUMAN_EVAL))
executor = ThreadPoolExecutor(max_workers=10)

# fmt: off
# list of task_ids that the baseline generated samples failed to solve even once
HARD_PROBLEMS = ['HumanEval/33', 'HumanEval/26', 'HumanEval/103', 'HumanEval/108', 'HumanEval/115', 'HumanEval/119', 'HumanEval/125', 'HumanEval/126', 'HumanEval/127', 'HumanEval/128', 'HumanEval/129', 'HumanEval/130', 'HumanEval/132', 'HumanEval/134', 'HumanEval/137', 'HumanEval/138', 'HumanEval/140', 'HumanEval/141', 'HumanEval/142', 'HumanEval/144', 'HumanEval/145', 'HumanEval/147', 'HumanEval/151', 'HumanEval/160', 'HumanEval/163', 'HumanEval/46', 'HumanEval/54', 'HumanEval/67', 'HumanEval/68', 'HumanEval/83', 'HumanEval/85', 'HumanEval/91', 'HumanEval/93', 'HumanEval/95', 'HumanEval/97']
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
    thread_problems = [{**problem, "test": test} for test in split_tests]
    results = []
    for result in executor.map(
        lambda tp: check_correctness(tp, completion, timeout), thread_problems
    ):
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
