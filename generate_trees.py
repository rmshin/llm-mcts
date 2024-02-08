import json

depth = 5
# decision tree
bf = 2
graph_dict = {0: {"nodes": [{"data": {"id": 0}}], "edges": []}}
for d in range(1, depth):
    last_nodes = graph_dict[d - 1]["nodes"]
    last_edges = graph_dict[d - 1]["edges"]
    offset = len(last_nodes) - bf ** (d - 1)
    parents = last_nodes[offset:]
    nodes = last_nodes.copy()
    edges = last_edges.copy()
    for i, p in enumerate(parents):
        base_idx = len(last_nodes) + i * bf
        child_nodes = [{"data": {"id": base_idx + ci}} for ci in range(bf)]
        nodes += child_nodes
        edges += [
            {
                "data": {
                    "source": p["data"]["id"],
                    "target": c["data"]["id"],
                    "selectable": False,
                }
            }
            for c in child_nodes
        ]
    graph_dict[d] = {"nodes": nodes, "edges": edges}

with open(f"web/public/graph_example_tree.json", "w") as f:
    json.dump(graph_dict, f)

# branching factor
# {[bf]: {"nodes": [...], "edges": [...]}}
graph_dict = {}
for bf in range(2, 6):
    idx = 1
    nodes, edges = [{"data": {"id": 0}}], []
    while idx < depth:
        offset = len(nodes) - bf ** (idx - 1)
        parents = nodes[offset:]
        for i, p in enumerate(parents):
            child_nodes = [{"data": {"id": len(nodes) + ci}} for ci in range(bf)]
            nodes += child_nodes
            edges += [
                {
                    "data": {
                        "source": p["data"]["id"],
                        "target": c["data"]["id"],
                        "selectable": False,
                    }
                }
                for c in child_nodes
            ]
        idx += 1
    graph_dict[bf] = {"nodes": nodes, "edges": edges}

with open(f"web/public/graph_branching_factor.json", "w") as f:
    json.dump(graph_dict, f)
