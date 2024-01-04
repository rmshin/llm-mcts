from graphviz import Digraph
from llama_cpp import Llama
from math import exp, log, inf, sqrt
import random

model = Llama(
    model_path="../llama.cpp/models/deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=256,
    n_threads=10,
    logits_all=True,
)

max_rollouts = 64
top_k = 3
beam_width = 1
problem_description = """Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True"""
# cache of generated programs => rewards
program_dict = {}
# hyperparameters for P-UCB function
c_base = 10
c = 4


class Node:
    def __init__(self, label, logprob, state, parent):
        self.value = 0
        self.label = label  # token label text for visualisation
        self.prob = exp(logprob)  # necessary for P-UCB calculation
        self.state = state  # full generated text
        self._children = []
        self._parent = parent
        self.visits = 0

    def backprop(self, value):
        if value > self.value:
            self.value = value
            if self._parent is not None:
                self._parent.backprop(value)


# Implements P-UCB heuristic as defined in https://arxiv.org/pdf/2303.05510.pdf#subsection.D.1
# P-UCB-SELECT(s, c) = argmax_a P-UCB(s, a)
# -> where P-UCB(s, a) = Q(s, a) + ß(s) * P(a|s) * √log(s.visits) / (1 + s'.visits)
# -> where ß(s) = log((s.visits + c_base + 1) / c_base) + c
# -> c_base & c are hyperparameters, set to values c_base = 10 & c = 4
def p_ucb_select(parent_node, child_nodes):
    Q = parent_node.value
    s_visits = parent_node.visits
    beta = log((s_visits + c_base + 1) / c_base) + c

    max_p_ucb = -inf
    max_node = None
    for i in range(len(child_nodes)):
        node = child_nodes[i]
        p_ucb = Q + beta * node.prob * sqrt(log(s_visits)) / (1 + node.visits)
        if p_ucb > max_p_ucb:
            max_node = child_nodes[i]
    return max_node


def get_top_k_tokens(curr_node, k):
    output = model(prompt=curr_node.state, max_tokens=1, temperature=0.2, logprobs=k)
    output_probs = output["choices"][0]["logprobs"]["top_logprobs"][0]
    return output_probs.items()


# TODO: use HF beam search/generate API for this part
def beam_search(curr_node):
    # output = model(
    #     prompt=curr_node.state, max_tokens=-1, temperature=0.2, top_k=beam_width
    # )
    # output_text = output["choices"][0]["text"]
    # return output_text[
    #     len(problem_description) :
    # ].strip()  # ignore original problem description in prefix
    return ""


# TODO: implement proper reward function
def calculate_reward(program_text):
    return random.random()


root = Node("<PD>", log(1), problem_description, None)
for i in range(max_rollouts):
    curr_node = root
    curr_node.visits += 1
    # selection
    while len(curr_node._children) > 0:
        curr_node = p_ucb_select(curr_node, curr_node._children)
        curr_node.visits += 1

    # expansion
    tokens = get_top_k_tokens(curr_node, top_k)
    child_nodes = [
        Node(token, logprob, curr_node.state + token, curr_node)
        for (token, logprob) in tokens
    ]
    curr_node._children = child_nodes

    # evaluation
    # TODO: fix beam search
    generated_program = beam_search(curr_node)
    reward = calculate_reward(generated_program)

    # backprop
    curr_node.backprop(reward)

    # TODO: early termination if fully accurate program has been generated
    if reward == 1:
        pass


##### Visualisation #####


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "TB"})  # TB = top to bottom
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | reward %.4f | visits %d | prob %.4f }"
            % (
                # HACK: quick-fix to escape special chars for label processing
                repr(n.label)
                .replace(">", "\>")
                .replace("<", "\<")
                .replace("}", "\}")
                .replace("{", "\{"),
                n.value,
                n.visits,
                n.prob,
            ),
            shape="record",
        )
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n2)), str(id(n1)))
    return dot


dot = draw_dot(root)
dot.render(filename="tree", view=True)
