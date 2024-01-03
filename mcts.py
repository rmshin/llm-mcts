from graphviz import Digraph
from llama_cpp import Llama
import random

model = Llama(
    model_path="deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=256,
    n_threads=10,
)

max_rollouts = 512
explore_c = 1e-2
top_k = 3
beam_width = 1
problem_description = """Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True"""
# cache of generated programs => rewards
program_dict = {}


class Node:
    def __init__(self, text, parent):
        self.value = 0
        self.text = text
        self._children = []
        self._parent = parent
        self.visits = 0

    def backprop(self, value):
        if value > self.value:
            self.value = value
            if self._parent is not None:
                self._parent.backprop(value)


def p_ucb_select(nodes):
    # start with randomly selected idx
    max_idx = random.randint(0, len(nodes))
    for i in range(len(nodes)):
        if nodes[i].value > nodes[max_idx].value:
            max_idx = i
    return nodes[max_idx]


def generate_next_n_tokens(model, input, n_tokens=-1):
    output = model(prompt=input, max_tokens=n_tokens, temperature=0.2, top_k=beam_width)
    output_text = output["choices"][0]["text"].strip()
    return output_text


def get_top_k_texts(curr_node, k):
    texts = []
    for _ in range(k):
        output = generate_next_n_tokens(model, curr_node.text, 1)
        texts.append(output)
    return texts


def beam_search(curr_node):
    output_text = generate_next_n_tokens(model, curr_node.text)
    return output_text[
        len(problem_description) :
    ].strip()  # ignore original problem description in prefix


# TODO: implement proper reward function
def calculate_reward(program_text):
    return random.random()


root = Node(problem_description, None)
for i in range(max_rollouts):
    curr_node = root
    curr_node.visits += 1
    # selection
    while len(curr_node._children) > 0:
        curr_node = p_ucb_select(curr_node._children)
        curr_node.visits += 1

    # expansion
    texts = get_top_k_texts(curr_node, top_k)
    child_nodes = [Node(text, curr_node) for text in texts]
    curr_node._children = child_nodes

    # evaluation
    generated_program = beam_search(curr_node)
    reward = calculate_reward(generated_program)

    # backprop
    curr_node.backprop(reward)


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
            label="{ %s | reward %.4f }" % (n.text[-1], n.value),
            shape="record",
        )
    # TODO: finish rendering edges
    return dot
