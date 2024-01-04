from graphviz import Digraph
from llama_cpp import Llama
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
explore_c = 1e-2
top_k = 3
beam_width = 1
problem_description = """Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True"""
# cache of generated programs => rewards
program_dict = {}


class Node:
    def __init__(self, label, state, parent):
        self.value = 0
        self.label = label  # token label text for visualisation
        self.state = state  # full generated text
        self._children = []
        self._parent = parent
        self.visits = 0

    def backprop(self, value):
        if value > self.value:
            self.value = value
            if self._parent is not None:
                self._parent.backprop(value)


# TODO: implement proper p-ucb function
def p_ucb_select(nodes):
    max_node = nodes[0]
    for i in range(len(nodes)):
        if nodes[i].value > max_node.value:
            max_node = nodes[i]
    return max_node


def get_top_k_tokens(curr_node, k):
    output = model(prompt=curr_node.state, max_tokens=1, temperature=0.2, logprobs=k)
    output_probs = output["choices"][0]["logprobs"]["top_logprobs"][0]
    return output_probs.keys()


def generate_next_n_tokens(model, input, n_tokens=-1):
    output = model(prompt=input, max_tokens=n_tokens, temperature=0.2, top_k=beam_width)
    output_text = output["choices"][0]["text"]
    return output_text


# TODO: use HF beam search/generate API for this part
def beam_search(curr_node):
    output_text = generate_next_n_tokens(model, curr_node.state)
    return output_text[
        len(problem_description) :
    ].strip()  # ignore original problem description in prefix


# TODO: implement proper reward function
def calculate_reward(program_text):
    return random.random()


root = Node("PD", problem_description, None)
for i in range(max_rollouts):
    curr_node = root
    curr_node.visits += 1
    # selection
    while len(curr_node._children) > 0:
        curr_node = p_ucb_select(curr_node._children)
        curr_node.visits += 1

    # expansion
    tokens = get_top_k_tokens(curr_node, top_k)
    child_nodes = [Node(token, curr_node.state + token, curr_node) for token in tokens]
    curr_node._children = child_nodes

    # evaluation
    # TODO: fix beam search
    # generated_program = beam_search(curr_node)
    generated_program = ""
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
            label="{ %s | reward %.4f | visits %d }"
            % (
                # HACK: quick-fix to escape special chars for label processing
                repr(n.label)
                .replace(">", "\>")
                .replace("<", "\<")
                .replace("}", "\}")
                .replace("{", "\{"),
                n.value,
                n.visits,
            ),
            shape="record",
        )
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n2)), str(id(n1)))
    return dot


dot = draw_dot(root)
dot.render(filename="tree", view=True)
