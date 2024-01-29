from graphviz import Digraph, escape


def trace_graph(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child.id, v.id))
                build(child)

    build(root)
    return nodes, edges


def draw_graphviz_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "TB"})  # TB = top to bottom
    nodes, edges = trace_graph(root)
    for n in nodes:
        uid = str(n.id)
        # HACK: quick-fix to escape special '<' & '>' chars during graphviz label processing
        escaped_n_label = (
            escape(n.label)
            .replace(">", "\>")
            .replace("<", "\<")
            .replace("}", "\}")
            .replace("{", "\{")
        )
        label = f"{{ {uid} | {escaped_n_label} | value {n.value:.4f} | p_ucb {n.p_ucb:.4f} | visits {n.visits} | prob {n.prob:.4f} }}"
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label=label,
            shape="record",
        )
    for id1, id2 in edges:
        dot.edge(str(id2), str(id1))
    return dot


def render_graphviz_tree(root, filename="tree", view=True):
    dot = draw_graphviz_dot(root)
    dot.render(filename=filename, view=view)
