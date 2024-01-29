from visualise import render_graphviz_tree
from llama_cpp import Llama
from math import exp, log, inf, sqrt
import time, sys
import itertools, json, copy


def main():
    outfile_prefix = ""
    if (len(sys.argv) == 2) and sys.argv[1] == "verilog":
        from verilogeval import (
            stats_execute,
            get_prompts_with_ids,
            STOP_SEQUENCES,
        )
        from verilog_eval.data import write_jsonl

        outfile_prefix = "verilog_"
    else:
        from humaneval import stats_execute, get_prompts_with_ids, STOP_SEQUENCES
        from human_eval.data import write_jsonl

    model = Llama(
        model_path="deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=1024,
        n_threads=10,
        logits_all=True,
    )

    max_rollouts = 128
    top_k = 3
    beam_width = 1
    # hyperparameters for P-UCB function
    c_base = 10
    c = 4

    class Node:
        id_iter = itertools.count()

        def __init__(self, label, logprob, state, parent):
            self.value = 0  # total reward obtainable from node
            self.prob = exp(logprob)  # necessary for P-UCB calculation
            self.state = state  # full generated text
            self._children = []
            self._parent = parent
            self.visits = 0
            ### attributes for graph visualisation
            self.id = next(self.id_iter)
            self.label = repr(label)  # token label text
            self.p_ucb = 0  # last calculated p_ucb value

        def backprop(self, value):
            # only propagate if new reward is greater than current max
            if value > self.value:
                self.value = value
                if self._parent is not None:
                    self._parent.backprop(value)

    class NodeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Node):
                cpy = copy.copy(obj)
                del cpy._parent
                del cpy._children
                return vars(cpy)
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)

    # Implements P-UCB heuristic as defined in https://arxiv.org/pdf/2303.05510.pdf#subsection.D.1
    # P-UCB-SELECT(s, c) = argmax_a P-UCB(s, a)
    # -> where P-UCB(s, a) = Q(s, a) + ß(s) * P(a|s) * √log(s.visits) / (1 + s'.visits)
    # -> where ß(s) = log((s.visits + c_base + 1) / c_base) + c
    # -> c_base & c are hyperparameters, set to values c_base = 10 & c = 4
    def p_ucb_select(parent_node, child_nodes):
        s_visits = parent_node.visits
        beta = log((s_visits + c_base + 1) / c_base) + c

        max_p_ucb = -inf
        max_node = None
        for i in range(len(child_nodes)):
            node = child_nodes[i]
            p_ucb = node.value + beta * node.prob * sqrt(log(s_visits)) / (
                1 + node.visits
            )
            node.p_ucb = p_ucb  # store most recent p_ucb for visualisation
            if p_ucb > max_p_ucb:
                max_node = node
                max_p_ucb = p_ucb
        return max_node

    def get_top_k_tokens(curr_node, k):
        output = model(prompt=curr_node.state, max_tokens=1, temperature=1, logprobs=k)
        output_probs = output["choices"][0]["logprobs"]["top_logprobs"][0]
        return output_probs.items()

    # TODO: update once https://github.com/abetlen/llama-cpp-python/pull/631 lands.
    # Right now we perform a simple greedy search as long as beam_width = 1. If we want to
    # support beam_width > 1 before the linked PR is complete we will have to be implement
    # the function using the low-level llama-cpp-python APIs.
    def beam_search(curr_node):
        """
        Returns the full generation with both prompt + completion concatenated.
        Original prompt needs to be indexed out to get the actual generated program.
        """
        output = model(
            prompt=curr_node.state,
            max_tokens=1024,
            temperature=0.2,
            top_k=beam_width,
            stop=STOP_SEQUENCES,
        )
        output_text = output["choices"][0]["text"]
        return curr_node.state + output_text

    def calculate_reward(task_id, completion):
        stats = stats_execute(task_id, completion)
        return stats["pass_rate"]

    # check if a generated program exists for a given node state and return reward if found
    def match_cached_programs(prefix, program_dict):
        for program, reward in program_dict.items():
            if program.startswith(prefix):
                return reward
        return -1

    def get_best_program(program_dict):
        max_reward = -inf
        best_program = None
        for program, reward in program_dict.items():
            if reward > max_reward:
                best_program = program
                reward = max_reward
        return best_program

    prompts_ids = get_prompts_with_ids()
    start = time.perf_counter()
    num_iter = 1
    for prompt, task_id in prompts_ids:
        prompt_start = time.perf_counter()
        print(f"---- STARTING MCTS FOR {task_id} ({num_iter}/{len(prompts_ids)}) ----")
        # cache of generated programs => rewards
        program_dict = {}
        num_rollouts = max_rollouts
        root = Node("<PD>", log(1), prompt, None)
        test_times = []
        # graph snapshots for web visualisation
        nodes, edges = {root.id: root}, {}
        graph_dict = {}
        for i in range(max_rollouts):
            graph_dict[i] = {
                "selectedNodes": [root.id],
                "state": "",
                "completion": "",
                "reward": 0,
                "task_id": task_id,
            }
            curr_node = root
            curr_node.visits += 1
            # selection
            while len(curr_node._children) > 0:
                for child in curr_node._children:
                    nodes[child.id] = child
                    edges[(curr_node.id, child.id)] = True
                curr_node = p_ucb_select(curr_node, curr_node._children)
                graph_dict[i]["selectedNodes"].append(curr_node.id)
                curr_node.visits += 1

            # expansion
            tokens = get_top_k_tokens(curr_node, top_k)
            child_nodes = [
                Node(token, logprob, curr_node.state + token, curr_node)
                for (token, logprob) in tokens
            ]
            curr_node._children = child_nodes
            for child in child_nodes:
                nodes[child.id] = child
                edges[(curr_node.id, child.id)] = True

            # evaluation
            reward = match_cached_programs(curr_node.state, program_dict)
            # only run generation if node state not found in cached programs
            if reward == -1:
                generated_program = beam_search(curr_node)
                completion = generated_program.replace(prompt, "")
                test_start = time.perf_counter()
                reward = calculate_reward(task_id, completion)
                test_end = time.perf_counter()
                test_times.append(test_end - test_start)
                program_dict[generated_program] = reward
                graph_dict[i]["state"] = curr_node.state.replace(prompt, "")
                graph_dict[i]["completion"] = completion
                graph_dict[i]["reward"] = reward
            graph_dict[i]["nodes"] = list(nodes.values())
            graph_dict[i]["edges"] = list(edges.keys())

            # backprop
            curr_node.backprop(reward)

            if reward == 1:
                num_rollouts = i + 1
                break
        best_completion = get_best_program(program_dict).replace(prompt, "")
        end = time.perf_counter()
        item = dict(
            task_id=task_id,
            completion=best_completion,
            stats=dict(
                num_rollouts=num_rollouts,
                num_generations=len(program_dict.keys()),
                eval_time=f"{(end - prompt_start):.4f}s",
                mean_test_time=f"{(sum(test_times)/len(test_times)):.4f}s",
            ),
        )
        write_jsonl(f"{outfile_prefix}few_shot_mcts.jsonl", [item], append=True)
        print(f"---- COMPLETED MCTS FOR {task_id} ({num_iter}/{len(prompts_ids)}) ----")
        print(f"Eval time: {(end - prompt_start):.4f}s")
        print(f"Mean test time: {(sum(test_times)/len(test_times)):.4f}s")
        print(f"Stats: {item['stats']}")
        num_iter += 1
        render_graphviz_tree(
            root, filename=f"svgviz/tree_{task_id.replace('/', '_')}", view=False
        )
        with open(f"web/public/graph_{task_id.replace('/', '_')}.json", "w") as f:
            json.dump(graph_dict, f, cls=NodeEncoder)

    end = time.perf_counter()
    print(f"Total elapsed time: {(end - start):.4f}s\n")


# necessary to prevent multiple executions of main() within stats_execute threads
if __name__ == "__main__":
    main()
