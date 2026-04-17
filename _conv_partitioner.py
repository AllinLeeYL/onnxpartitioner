import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, TensorProto
# import onnxruntime as ort
import numpy as np
# import onnx_graphsurgeon as gs
from _buffer import Buffer
from common import remove_nodes, get_value_info


def max_multiplier_within_limit(base: int, limit: int) -> int:
    return 0 if base == 0 else limit // base


def conv_params(graph, node):
    value_info = get_value_info(graph)
    In_shape = value_info[node.input[0]]
    W_shape = value_info[node.input[1]]
    Out_shape = value_info[node.output[0]]
    strides = next((list(attr.ints) for attr in node.attribute if attr.name == "strides"), [1, 1])
    pads = next((list(attr.ints) for attr in node.attribute if attr.name == "pads"), [1, 1])
    return [In_shape, W_shape, Out_shape, strides, pads]


def calculate_conv_buf(graph, node):
    In_shape, W_shape, Out_shape, _, _ = conv_params(graph, node)
    # Input
    N, C, H, W = In_shape
    in_buf = Buffer(C, H*W)

    # Weight
    # W_name = node.input[1]
    # W = self._init_map[W_name]
    # out_c, in_c, k_h, k_w = W.shape

    # Output
    N, C, H, W = Out_shape
    out_buf = Buffer(C, H*W)
    return in_buf, Buffer(0, 0), out_buf


def parition_conv(graph, node, hardware):
    in_buf, w_buf, out_buf = calculate_conv_buf(graph, node)

    In_shape, W_shape, Out_shape, strides, pads = conv_params(graph, node)
    _, _, k_h, k_w = W_shape
    iN, iC, iH, iW = In_shape
    oN, oC, oH, oW = Out_shape
    stride = strides[0]

    In_name, W_name = node.input[0:2]
    Out_name = node.output[0]

    # some checks
    assert k_h == k_w
    # assert iH == iW
    # assert oH == oW
    assert iN == 1
    assert oN == 1
    assert strides[0] == strides[1]

    if out_buf.channel_s > hardware['output_buffer'].channel_s or in_buf.channel_s > hardware['input_buffer'].channel_s:
        pass
    else: # pixel_s > hardware.pixel_s
        # divide by input
        i_h = max_multiplier_within_limit(base=iW, limit=hardware['input_buffer'].pixel_s - iW*(k_h-1))
        o_h_by_input = i_h // stride
        # divide by output
        o_h = max_multiplier_within_limit(base=oW, limit=hardware['output_buffer'].pixel_s) 
        o_h = min(o_h, o_h_by_input)
        q, r = divmod(oH, o_h)
        # assert(r==0)
        n_seg = q if r==0 else q+1
        # insert slice node
        for i in range(0, n_seg):
            slice_node = helper.make_node(
                "Slice",
                inputs=[In_name, 
                        In_name+'_slice_starts_'+str(i),
                        In_name+'_slice_ends_'+str(i),
                        In_name+'_slice_axes_'+str(i)
                ],
                outputs=[In_name+'_slice_'+str(i)],
            )
            graph.node.append(slice_node)
            starts = max(o_h*stride*i-k_h+1, 0)
            ends = min(o_h*stride*(i+1)+k_h-1, iH)
            starts_tensor_info = helper.make_tensor_value_info(
                In_name+'_slice_starts_'+str(i),
                TensorProto.FLOAT,
                [0, 0, starts, 0]
            )
            ends_tensor_info = helper.make_tensor_value_info(
                In_name+'_slice_ends_'+str(i),
                TensorProto.FLOAT,
                [iN, iC, ends, iW]
            )
            axes_tensor_info = helper.make_tensor_value_info(
                In_name+'_slice_axes'+str(i),
                TensorProto.FLOAT,
                [0, 1, 2, 3]
            )
            graph.value_info.append(starts_tensor_info)
            graph.value_info.append(ends_tensor_info)
            graph.value_info.append(axes_tensor_info)
        # insert sub Conv node
        for i in range(n_seg):
            pad_top, pad_left, pad_bottom, pad_right = pads # top left bottom right
            pad_top = pad_top if i==0 else 0
            pad_bottom = pad_bottom if i==q-1 else 0
            conv_node = helper.make_node(
                "Conv",
                inputs=[In_name+'_slice_'+str(i), W_name],
                outputs=[node.name+'_out_'+str(i)],
                kernel_shape=[k_h, k_w],
                strides=strides,
                pads=[pad_top, pad_left, pad_bottom, pad_right]
            )
            graph.node.append(conv_node)
        # insert Concat node
        concat_node = helper.make_node(
            "Concat",
            inputs=[node.name+'_out_'+str(i) for i in range(n_seg)],
            outputs=[Out_name],
            axis=1
        )
        graph.node.append(concat_node)
        # insert values info
        for i in range(n_seg):
            starts = max(o_h*stride*i-(k_h-1)//2, 0)
            ends = min(o_h*stride*(i+1)+(k_h-1)//2, iH)
            o_h_t = o_h if i != n_seg-1 else oH - (i*o_h)
            in_tensor_info = helper.make_tensor_value_info(
                In_name+'_slice_'+str(i),
                TensorProto.FLOAT,
                [iN, iC, ends-starts, iW]
            )
            out_tensor_info = helper.make_tensor_value_info(
                node.name+'_out_'+str(i),
                TensorProto.FLOAT,
                [oN, oC, o_h_t, oW]
            )
            graph.value_info.append(in_tensor_info)
            graph.value_info.append(out_tensor_info)
        # insert init map
        # for i in range(q):
        #      = np.random.randn(16, 3, 3, 3).astype(np.float32)
        # remove original node
        remove_nodes(graph, lambda n: n == node)
    return graph
