import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, TensorProto
# import onnxruntime as ort
import argparse
import numpy as np
import onnx_graphsurgeon as gs

from _buffer import Buffer
from _conv_partitioner import parition_conv, calculate_conv_buf


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


    def _update(self):
        # update init_map
        self._init_map = {
            init.name: numpy_helper.to_array(init)
            for init in self._graph.initializer
        }


    def partition(self, graph):
        self._graph = graph
        self._update()
        while self._partition_run():
            pass
        return self._graph


    def _partition_run(self):
        partitioned = False
        for node in self._graph.node:
            if self._exceed_hardware_limit(node):
                self._split(node)
                partitioned = True
                break
        self._update()
        return partitioned


    def _exceed_hardware_limit(self, node):
        in_buf, weight_buf, out_buf = self._calculate_buf(node)
        return (
            in_buf > hardware['input_buffer'] or 
            out_buf > hardware['output_buffer']
        )


    def _calculate_buf(self, node):
        in_buf = Buffer(0, 0)
        w_buf = Buffer(0, 0)
        out_buf = Buffer(0, 0)
        if node.op_type == 'Conv':
            in_buf, w_buf, out_buf = calculate_conv_buf(self._graph, node)

        return in_buf, w_buf, out_buf


    def _split(self, node):
        if node.op_type == 'Conv':
            self._graph = parition_conv(self._graph, node, self.hardware)


# input: model + hardware parameters
# output: mapping
if __name__ == '__main__':
    args = parse_argument()

    if (args.model[-5:] != '.onnx'):
        print("model file not ended with \".onnx\" may raise errors.")
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    
    hardware = {'input_buffer': Buffer(256, 8192*8),
                'output_buffer': Buffer(256, 4096)}

    partitioner = Partitioner(hardware)
    partitioned_graph = partitioner.partition(model.graph)
    model = helper.make_model(partitioned_graph)
    onnx.save(model, args.model[:-5]+"_partitioned.onnx")

    # Model metadata
    # print("IR version:", model.ir_version)
    # print("Producer:", model.producer_name)
