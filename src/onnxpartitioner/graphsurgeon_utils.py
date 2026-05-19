
import onnx_graphsurgeon as gs

def get_parent_from_tensor(inp: gs.Tensor) -> gs.Node:
    assert(len(inp.inputs) == 1)
    return inp.inputs[0]

def get_parents(node: gs.Node) -> list[gs.Node]:
    parents = []
    for in_tensor in node.inputs:
        assert(len(in_tensor.inputs) == 1 or len(in_tensor.inputs) == 0)
        for parent_node in in_tensor.inputs:
            parents.append(parent_node)
    return parents

def get_successors(node: gs.Node) -> list[gs.Node]:
    successors = []
    for out_tensor in node.outputs:
        for child_node in out_tensor.outputs:
            successors.append(child_node)
    return successors

def remove_node(graph: gs.Graph, node: gs.Node):
    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.remove(node)

@gs.Graph.register()
def add_node(self, op, inputs, outputs, **kwargs):
    return self.layer(op=op, inputs=inputs, outputs=outputs, **kwargs)


@gs.Graph.register()
def remove_node(self, node: gs.Node):
    node.inputs.clear()
    node.outputs.clear()
    self.nodes.remove(node)