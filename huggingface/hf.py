from humaneval import get_prompts_with_ids
from human_eval.data import write_jsonl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from tqdm import tqdm
import sys

outfile_prefix = "hf_humaneval_"
if (len(sys.argv) == 2) and sys.argv[1] == "verilog":
    from verilogeval import get_prompts_with_ids
    from verilog_eval.data import write_jsonl

    outfile_prefix = "verilog_"
else:
    from humaneval import get_prompts_with_ids
    from human_eval.data import write_jsonl


model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, temperature=1., trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_name, temperature=1., trust_remote_code=True, load_in_8bit=True, device_map='auto')
model = torch.compile(model)

N_SAMPLES = 5
prompts_ids = get_prompts_with_ids()[20:36]
for prompt, task_id in tqdm(prompts_ids):
    print(task_id)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    samples = []
    
    for _ in tqdm(range(4)):
        output = model.generate(input_ids, max_new_tokens=250, do_sample=True, top_k=50, num_return_sequences=N_SAMPLES, pad_token_id=tokenizer.eos_token_id)
        output_str = tokenizer.batch_decode(output[:, input_ids.shape[-1]:], skip_special_tokens=True)
        output_clean_str = [x.split("```")[0].rstrip() for x in output_str]
        samples.extend([dict(task_id=task_id, completion=res) for res in output_clean_str])
        write_jsonl(
            f"{outfile_prefix}few_shot_baselines_250_top_50.jsonl", samples, append=True
        )

# run this to get the pass@k metrics
# evaluate_functional_correctness samples.jsonl
