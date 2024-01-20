import outlines
from interpreter import evaluate_code
from verilog_eval.data import read_problems
from verilog_eval.data import VERILOG_EVAL_HUMAN, HUMAN_DESCRIPTIONS

problems = read_problems(VERILOG_EVAL_HUMAN)
descriptions = read_problems(HUMAN_DESCRIPTIONS)
for task_id, item in descriptions.items():
    problems[task_id]["description"] = item["detail_description"]

task_id_problem_map = problems.copy()

problems = list(problems.values())

STOP_SEQUENCES = ["```"]


def stats_execute(task_id, completion, timeout=10):
    problem = task_id_problem_map[task_id]
    res = evaluate_code(task_id, completion, problem)
    return {"task_id": task_id, "pass_rate": res.pass_rate}


@outlines.prompt
def few_shot_prompt(instructions, examples, description, question):
    """{{ instructions }}

    {% for example in examples %}
    Description:
    ```
    {{ example.description }}
    ```
    Question:
    ```
    {{ example.prompt }}
    ```
    Answer:
    ```
    {{ example.canonical_solution }}
    ```
    {% endfor %}

    Description:
    ```
    {{ description }}
    ```
    Question:
    ```
    {{ question }}
    ```
    Answer:
    ```
    """


instructions = "Please answer the following question following the examples. Generate valid verilog code always."
examples = problems[:2]


def get_prompts_with_ids():
    prompts_with_ids = [
        (
            few_shot_prompt(
                instructions, examples, problem["description"], problem["prompt"]
            ),
            problem["task_id"],
        )
        for problem in problems[2:]
    ]
    return prompts_with_ids
