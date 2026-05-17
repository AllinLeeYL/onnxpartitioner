
import onnx_graphsurgeon as gs

def get_parents(node: gs.Node) -> list[gs.Node]:
    parents = []
    for out_tensor in node.inputs:
        for parent_node in out_tensor.inputs:
            parents.append(parent_node)
    return parents

def get_successors(node: gs.Node) -> list[gs.Node]:
    successors = []
    for out_tensor in node.outputs:
        for child_node in out_tensor.outputs:
            successors.append(child_node)
    return successors