from dataclasses import dataclass

@dataclass
class Buffer:
    channel_s: int
    pixel_s: int

    def __gt__(self, other):
        return self.channel_s > other.channel_s or self.pixel_s > other.pixel_s


@dataclass
class ConvSpec:
    # Input
    in_h: int
    in_w: int
    in_name: str
    in_channel: int

    # Output
    out_h: int
    out_w: int
    out_name: str
    out_channel: int

    # kernel
    k_h: int
    k_w: int
    k_name: str

    # bias
    b_name: str

    # other params
    pads: tuple
    strides: tuple
    batch: int


def get_value_info(graph):
    values = list(graph.value_info) + list(graph.input) + list(graph.output)
    value_info = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in values
    }
    return value_info


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



def max_multiplier_within_limit(base: int, limit: int) -> int:
    return 0 if base == 0 else limit // base