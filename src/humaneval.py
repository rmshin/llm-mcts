import outlines
from concurrent.futures import ThreadPoolExecutor
from human_eval.execution import check_correctness
from human_eval.data import HUMAN_EVAL, stream_jsonl

problems = list(stream_jsonl(HUMAN_EVAL))

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
    # NOTE: assumes human-eval-specific logic for multiline asserts & for-loops
    # won't work properly if multiline assert nested within for-loop
    multiline_assert, parts = False, []
    for_loop, fl_parts = False, []
    for i in tests.split("\n"):
        if multiline_assert:
            parts.append(i)
            if i.lstrip().startswith("]"):
                test = "\n".join(parts)
                split_tests.append(pre_base_str + base_str + test)
                multiline_assert = False
        elif i.lstrip().startswith("assert") and i.lstrip()[-1] == "[":
            multiline_assert = True
            parts = [i]
        elif for_loop:
            fl_parts.append(i)
            if i.lstrip().startswith("assert"):
                test = "\n".join(fl_parts)
                split_tests.append(pre_base_str + base_str + test)
                for_loop, fl_parts = False, []
        elif (
            (i.lstrip() == "")
            or (i.lstrip().startswith("#"))
            or (i.lstrip().startswith("print"))
        ):
            continue
        elif not (i.lstrip().startswith("assert")):
            fl_parts.append(i)
            if i.lstrip().startswith("for"):
                for_loop = True
        # special logic for HumanEval/151
        elif problem["task_id"] == "HumanEval/151" and (
            i.lstrip().startswith("assert candidate(lst)")
        ):
            fl_parts.append(i)
            test = "\n".join(fl_parts)
            split_tests.append(pre_base_str + base_str + test)
            fl_parts = []
        else:
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
    with ThreadPoolExecutor() as executor:
        for result in executor.map(
            lambda tp: check_correctness(tp, completion, timeout), thread_problems
        ):
            results.append(result["passed"])

    return {
        "task_id": task_id,
        "pass_rate": sum(results) / len(results),
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


def get_skip_prompts_with_ids():
    prompts_with_ids = [
        (few_shot_prompt(instructions, examples, problem["prompt"]), problem["task_id"])
        for problem in problems[2:]
        if problem["task_id"] in SKIP_PROBLEM_IDS
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
