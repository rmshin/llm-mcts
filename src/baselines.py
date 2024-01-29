from humaneval import get_prompts_with_ids
from human_eval.data import write_jsonl
from llama_cpp import Llama
from tqdm import tqdm
import sys

outfile_prefix = ""
if (len(sys.argv) == 2) and sys.argv[1] == "verilog":
    from verilogeval import get_prompts_with_ids
    from verilog_eval.data import write_jsonl

    outfile_prefix = "verilog_"
else:
    from humaneval import get_prompts_with_ids
    from human_eval.data import write_jsonl

model = Llama(
    model_path="deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    n_batch=1024,
    n_threads=8,
)

N_SAMPLES = 20
prompts_ids = get_prompts_with_ids()
for prompt, task_id in tqdm(prompts_ids):
    samples = []
    # TODO: update to generate in parallel once https://github.com/abetlen/llama-cpp-python/pull/951 lands.
    for _ in tqdm(range(N_SAMPLES)):
        output = model(
            prompt=prompt, max_tokens=1024, temperature=1, top_k=50, stop=["```"]
        )
        res = output["choices"][0]["text"]
        item = dict(task_id=task_id, completion=res)
        samples.append(item)
    write_jsonl(
        f"{outfile_prefix}few_shot_baselines_1024_top_50.jsonl", samples, append=True
    )

# run this to get the pass@k metrics
# evaluate_functional_correctness samples.jsonl
