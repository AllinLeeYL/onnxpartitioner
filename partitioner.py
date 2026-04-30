import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, version_converter
# import onnxruntime as ort
import argparse
import numpy as np
# import onnx_graphsurgeon as gs

from common import Buffer
from _conv_partitioner import conv_exceed_hardware_limit, partition_conv


def parse_argument():
    parser = argparse.ArgumentParser(description='ONNX model partitioner')
    parser.add_argument('model', type=str,  
                        help='path to model.pt file.')
    parser.add_argument('--in_buffer_channel', type=int,
                        help='input buffer channel size')
    parser.add_argument('--in_buffer_pixel', type=int,
                        help='input buffer pixel size')
    parser.add_argument('--out_buffer_channel', type=int,
                        help='output buffer channel size')
    parser.add_argument('--out_buffer_pixel', type=int,
                        help='output buffer pixel size')
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
    def __init__(self, hardware):
        self.hardware = hardware
        self._graph = None # graph to be partitioned
        pass


    def partition(self, model):
        self._model = model
        self._graph = model.graph
        while self._partition_run():
            model = helper.make_model(self._graph)
            self._model = shape_inference.infer_shapes(model)
            self._graph = self._model.graph
            pass
        return self._model


    def _partition_run(self):
        partitioned = False
        for node in self._graph.node:
            if self._exceed_hardware_limit(node):
                self._split(node)
                partitioned = True
                break
        return partitioned


    def _exceed_hardware_limit(self, node):
        if node.op_type == 'Conv':
            return conv_exceed_hardware_limit(self._graph, node, hardware)
            
        return False


    def _split(self, node):
        if node.op_type == 'Conv':
            self._graph = partition_conv(self._graph, node, self.hardware)


# input: model + hardware parameters
# output: mapping
if __name__ == '__main__':
    args = parse_argument()

    if (args.model[-5:] != '.onnx'):
        print("model file not ended with \".onnx\" may raise errors.")
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    # model = shape_inference.infer_shapes(model)
    
    hardware = {'input_buffer': Buffer(256, 8192),
                'output_buffer': Buffer(256, 4096)}

    partitioner = Partitioner(hardware)
    partitioned_model = partitioner.partition(model)
    # model = helper.make_model(partitioned_graph)
    converted_model = version_converter.convert_version(partitioned_model, 25)
    onnx.save(converted_model, args.model[:-5]+"_partitioned.onnx")

    # Model metadata
    print("IR version:", model.ir_version)
    # print(model.graph.initializer)
    # print("Producer:", model.producer_name)
