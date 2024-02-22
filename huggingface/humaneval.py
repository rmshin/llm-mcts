import outlines
from concurrent.futures import ThreadPoolExecutor
from human_eval.execution import check_correctness
from human_eval.data import HUMAN_EVAL, stream_jsonl

problems = list(stream_jsonl(HUMAN_EVAL))

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
