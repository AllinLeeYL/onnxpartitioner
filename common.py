
def get_value_info(graph):
    values = list(graph.value_info) + list(graph.input) + list(graph.output)
    value_info = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in values
    }
    return value_info


def remove_nodes(graph, condition_fn):
    
    nodes = [
        n for n in graph.node
        if not condition_fn(n)
    ]
    graph.ClearField('node')
    graph.node.extend(nodes)