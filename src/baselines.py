from humaneval import get_prompts_with_ids
from human_eval.data import write_jsonl
from llama_cpp import Llama
from tqdm import tqdm

model = Llama(
    model_path="deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=256,
    n_threads=8,
)

N_SAMPLES = 5
samples = []
prompts_ids = get_prompts_with_ids()
for prompt, task_id in tqdm(prompts_ids):
    # TODO: update to generate in parallel once https://github.com/abetlen/llama-cpp-python/pull/951 lands.
    for _ in tqdm(range(N_SAMPLES)):
        output = model(prompt=prompt, max_tokens=256, temperature=0.2, stop=["```"])
        res = output["choices"][0]["text"]
        item = dict(task_id=task_id, completion=res)
        samples.append(item)

write_jsonl("few_shot_baselines.jsonl", samples)

# run this to get the pass@k metrics
# evaluate_functional_correctness samples.jsonl
