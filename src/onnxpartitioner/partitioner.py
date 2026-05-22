import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, version_converter, ModelProto
import argparse
import numpy as np
import onnx_graphsurgeon as gs

from .common import Buffer
from .graphsurgeon_utils import get_successors
from ._conv_partitioner import ConvPartitionPlan, conv_params, default_partition_plan, apply_conv_partition


def parse_argument():
    parser = argparse.ArgumentParser(description='ONNX model partitioner')
    parser.add_argument('model', type=str,  
                        help='path to model.pt file.')
    parser.add_argument('--in_channel', type=int, default=256,
                        help='input buffer channel size')
    parser.add_argument('--in_pixel', type=int, default=1024*1024,
                        help='input buffer pixel size')
    parser.add_argument('--out_channel', type=int, default=256,
                        help='output buffer channel size')
    parser.add_argument('--out_pixel', type=int, default=4096,
                        help='output buffer pixel size')
    parser.add_argument('--direction', choices=['auto', 'vertical', 'horizontal'], default='auto',
                        help='partition direction of 2d array')
    parser.add_argument('--opset', type=int, default=23,
                        help='onnx opset version of the partitioned model')
    args = parser.parse_args()
    return args


# Build a lookup: tensor name → shape
def get_shape(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_value if d.dim_value > 0 else d.dim_param for d in dims]


def Gemm_node_params(graph, node):
    W_name = node.input[1]
    W = numpy_helper.to_array(
        next(init for init in graph.initializer if init.name == W_name)
    ) # W_out, W_in
    W = np.transpose(W)
    W_in, W_out = W.shape
    return [W_in, W_out], W


def MaxPool_node_params(graph, node):
    # TODO: support non-square kernel shape and strides. 
    # TODO: support pads, dilations, and ceil mode.
    kernel_size = [list(attr.ints) for attr in node.attribute if attr.name == 'kernel_shape'][0][0]
    stride = [list(attr.ints) for attr in node.attribute if attr.name == 'strides'][0][0]
    return [kernel_size, stride], None


def Reshape_node_params(graph, node):
    init_map = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    shape_tensor = init_map[node.input[1]]
    return shape_tensor.tolist(), None


class Partitioner:
    def __init__(self, hardware, **kwargs):
        """
        Initialize the graph partitioner.

        Parameters
        ----------
        hardware : dict
            Hardware constraints, capabilities, or resource budget
            used to guide graph partitioning.

        **kwargs : dict
        Optional configuration parameters.

            Supported keys:
            - conv_partition_plan_func : callable
                Function used to generate Conv partition plans.
            - direction : str
                Partition traversal direction (e.g. auto, vertical, horizontal).

        """
        self.hardware = hardware
        self.kwargs = kwargs

        # Optional partition direction
        self.direction = kwargs.get(
            "direction",
            "auto"
        )
        # Optional partition planning function for Conv operators
        self.conv_partition_plan_func = kwargs.get(
            "conv_partition_plan_func",
            None
        )
        pass


    def partition(self, model: ModelProto | gs.Graph) -> gs.Graph:
        """
        Initialize the graph partitioner.

        Parameters
        ----------
        model : ModelProto
            model to be partitioned

        Attributes
        ----------
        _graph : Graph | None
            graph format used in onnx_graphsurgeon.
        """
        if isinstance(model, ModelProto):
            self._graph: gs.Graph = gs.import_onnx(model)
        elif isinstance(model, gs.Graph):
            self._graph = model
        else:
            raise RuntimeError(f"unsupported model format: {type(model)}")
        
        self._graph.toposort()
        while self._partition_run():
            pass
        return self._graph


    def _partition_run(self):
        partitioned = False
        for node in self._graph.nodes:
            if self._partition_node(node):
                partitioned = True
                break
        return partitioned


    def _partition_node(self, node: gs.Graph):
        is_partitioned = False
        if node.op == "Conv":
            is_partitioned = self._try_partition_conv(node)
        return is_partitioned
    

    def _try_partition_conv(self, node):
        partition_plan = self.conv_partition_plan_func if self.conv_partition_plan_func != None else default_partition_plan
        # Step 1: extract parameters
        spec = conv_params(self._graph, node)

        # Step 2: compute plan
        plan = partition_plan(spec, self.hardware, self.direction)
        if plan == None:
            return False
        print("Partition plan:", plan, "applied to", node.name)

        # Step 3: apply transformation
        last_node = apply_conv_partition(self._graph, node, spec, plan)
        if not plan.concat_node or not plan.sum_node:
            self._derived_partition(last_node=last_node, last_plan=plan)

        return True
    

    def _derived_partition(self, last_node, last_plan):
        if last_plan.n_out_channel != 0:
            new_plan = ConvPartitionPlan(n_in_channel=last_plan.n_out_channel, in_channel_s=last_plan.out_channel_s, do_slice=False)
            for sub_node in get_successors(last_node):
                if sub_node.op == "Conv":
                    spec = conv_params(self._graph, sub_node)
                    print("Derived plan:", new_plan, "applied to", sub_node.name)
                    apply_conv_partition(self._graph, sub_node, spec, new_plan)
                
        else:
            raise RuntimeError(f"Not supported operation: removing Concat/Sum node when partitioning {last_node.name}")

# input: model + hardware parameters
# output: mapping
def main():
    args = parse_argument()

    if (args.model[-5:] != '.onnx'):
        print("model file not ended with \".onnx\" may raise errors.")
    model = onnx.load(args.model)
    onnx.checker.check_model(model)

    
    hardware = {'input_buffer': Buffer(args.in_channel, args.in_pixel),
                'output_buffer': Buffer(args.out_channel, args.out_pixel)}


    # ------------ Partition -------------
    partitioner = Partitioner(hardware, direction=args.direction)
    partitioned_graph = partitioner.partition(model)
    partitioned_model = gs.export_onnx(partitioned_graph)

    # ------------ Save model ------------
    converted_model = version_converter.convert_version(partitioned_model, args.opset)
    onnx.save(converted_model, args.model[:-5]+"_partitioned.onnx")

    # Model metadata
    print("IR version:", model.ir_version)


if __name__ == "__main__":
    main()