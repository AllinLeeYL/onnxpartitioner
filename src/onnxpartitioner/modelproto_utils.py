def remove_nodes(graph, condition_fn):
    """
    remove node in ONNX graph
    """
    nodes = [
        n for n in graph.node
        if not condition_fn(n)
    ]
    graph.ClearField('node')
    graph.node.extend(nodes)


def get_value_info(graph):
    values = list(graph.value_info) + list(graph.input) + list(graph.output)
    value_info = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in values
    }
    for init in graph.initializer:
        if init.name not in value_info:
            value_info[init.name]=list(init.dims)
    return value_info