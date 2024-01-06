from graphviz import Digraph, escape


def trace_graph(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    i = 0  # node id

    def build(v):
        nonlocal i
        if v not in nodes:
            nodes.add((v, i))
            i += 1
            for child in v._children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_graphviz_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "TB"})  # TB = top to bottom
    nodes, edges = trace_graph(root)
    for n, i in nodes:
        uid = str(id(n))
        # HACK: quick-fix to escape special '<' & '>' chars during graphviz label processing
        escaped_n_label = (
            escape(repr(n.label))
            .replace(">", "\>")
            .replace("<", "\<")
            .replace("}", "\}")
            .replace("{", "\{")
        )
        label = f"{{ {i} | {escaped_n_label} | {n.value:.4f} | {n.p_ucb:.4f} | {n.visits} | {n.prob:.4f} }}"
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label=label,
            shape="record",
        )
    for n1, n2 in edges:
        dot.edge(str(id(n2)), str(id(n1)))
    return dot


def render_graphviz_tree(root, filename="tree"):
    dot = draw_graphviz_dot(root)
    dot.render(filename=filename, view=True)
