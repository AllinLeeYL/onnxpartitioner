import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, version_converter, ModelProto
import argparse
import numpy as np
# import onnx_graphsurgeon as gs

from .common import Buffer
from ._conv_partitioner import try_partition_conv


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
    def __init__(self, hardware, direction):
        self.hardware = hardware
        self.direction = direction
        self._graph = None # graph to be partitioned

        # partition plan functions
        self.conv_partition_plan_func = None
        pass


    def partition(self, model: ModelProto):
        self._model = model
        self._graph = model.graph
        while self._partition_run():
            self._model = helper.make_model(self._graph)
            self._model = shape_inference.infer_shapes(self._model)
            self._graph = self._model.graph
        return self._model


    def _partition_run(self):
        partitioned = False
        for node in self._graph.node:
            if self._partition_node(node):
                partitioned = True
                break
        return partitioned


    # def _exceed_hardware_limit(self, node):
    #     if node.op_type == 'Conv':
    #         return conv_exceed_hardware_limit(self._graph, node, self.hardware)
            
    #     return False


    def _partition_node(self, node):
        is_partitioned = False
        if node.op_type == 'Conv':
            is_partitioned = try_partition_conv(self._graph, node, self.hardware, self.direction, self.conv_partition_plan_func)
        return is_partitioned

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
    partitioned_model = partitioner.partition(model)
    

    # ------------ Save model ------------
    converted_model = version_converter.convert_version(partitioned_model, 25)
    onnx.save(converted_model, args.model[:-5]+"_partitioned.onnx")

    # Model metadata
    print("IR version:", model.ir_version)
    # print(model.graph.initializer)
    # print("Producer:", model.producer_name)


if __name__ == "__main__":
    main()