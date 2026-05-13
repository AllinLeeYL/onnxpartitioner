
def get_parents(node):
    parents = []
    for out_tensor in node.inputs:
        for parent_node in out_tensor.inputs:
            parents.append(parent_node)
    return parents