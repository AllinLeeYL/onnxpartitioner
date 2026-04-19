import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, TensorProto
# import onnxruntime as ort
import numpy as np

from common import *
from dataclasses import dataclass


def conv_exceed_hardware_limit(graph, node, hardware):
    buffers = calculate_conv_buf(graph, node)
    for buf_name, buf, in hardware.items():
        if buf > hardware[buf_name]:
            return True
    
    return False


def partition_conv(graph, node, hardware):
    # Step 1: extract
    spec = conv_params(graph, node)

    # Step 2: compute plan
    plan = compute_partition_plan(spec, hardware)

    # Step 3: apply transformation
    apply_conv_partition(graph, node, spec, plan)

    return graph


@dataclass
class _PartitionPlan:
    n_o_h_seg: int # number of output height segment
    o_h: int


def conv_params(graph, node):
    in_name = node.input[0]
    out_name = node.output[0]
    kernel_name = node.input[1]
    value_info = get_value_info(graph)
    strides = next((list(attr.ints) for attr in node.attribute if attr.name == "strides"), [1, 1])
    pads = next((list(attr.ints) for attr in node.attribute if attr.name == "pads"), [1, 1, 1, 1])

    return ConvSpec(
        in_h=value_info[in_name][2],
        in_w=value_info[in_name][3],
        in_name=in_name,
        in_channel=value_info[in_name][1],

        out_h=value_info[out_name][2],
        out_w=value_info[out_name][3],
        out_name=out_name,
        out_channel=value_info[out_name][1],

        k_h=value_info[kernel_name][2],
        k_w=value_info[kernel_name][3],
        k_name=kernel_name,
        batch=value_info[out_name][0],
        strides=strides,
        pads=pads
    )


def calculate_conv_buf(graph, node):
    spec = conv_params(graph, node)

    return {'input_buffer': Buffer(spec.in_channel, spec.in_h * spec.in_w),
            'output_buffer': Buffer(spec.in_channel, spec.out_h * spec.out_w)}


def compute_partition_plan(spec: ConvSpec, hardware):
    if spec.out_channel > hardware['output_buffer'].channel_s or \
        spec.in_channel > hardware['input_buffer'].channel_s:
        pass
    else:
        # -------- Partition along height dimension --------

        # Compute max input tile height that fits in input buffer
        i_h = max_multiplier_within_limit(
            base=spec.in_w,
            limit=hardware['input_buffer'].pixel_s - spec.in_w * (spec.k_h - 1)
        )

        # Corresponding output height from input tiling
        o_h_by_input = i_h // spec.strides[0]

        # Compute max output tile height that fits in output buffer
        o_h = max_multiplier_within_limit(
            base=spec.out_w,
            limit=hardware['output_buffer'].pixel_s
        )

        # Final tile height constrained by both input and output limits
        o_h = min(o_h, o_h_by_input)

        # Number of segments needed to cover full output
        q, r = divmod(spec.out_h, o_h)
        n_seg = q if r == 0 else q + 1
        return _PartitionPlan(o_h=o_h, n_o_h_seg=n_seg)


def apply_conv_partition(graph, node, spec: ConvSpec, plan: _PartitionPlan):
    # -------- Insert Slice nodes (split input feature map) --------
    for i in range(0, plan.n_o_h_seg):
        slice_node = helper.make_node(
            "Slice",
            inputs=[
                spec.in_name,
                spec.in_name + '_slice_starts_' + str(i),
                spec.in_name + '_slice_ends_' + str(i),
                spec.in_name + '_slice_axes_' + str(i)
            ],
            outputs=[spec.in_name + '_slice_' + str(i)],
        )
        graph.node.append(slice_node)

        # Compute slice boundaries (include overlap for kernel)
        starts = max(plan.o_h * spec.strides[0] * i - spec.k_h + 1, 0)
        ends = min(plan.o_h * spec.strides[0] * (i + 1) + spec.k_h - 1, spec.in_h)

        # Add tensor metadata for slice parameters
        starts_tensor_info = helper.make_tensor_value_info(
            spec.in_name + '_slice_starts_' + str(i),
            TensorProto.FLOAT,
            [0, 0, starts, 0]
        )
        ends_tensor_info = helper.make_tensor_value_info(
            spec.in_name + '_slice_ends_' + str(i),
            TensorProto.FLOAT,
            [spec.batch, spec.in_channel, ends, spec.in_w]
        )
        axes_tensor_info = helper.make_tensor_value_info(
            spec.in_name + '_slice_axes' + str(i),
            TensorProto.FLOAT,
            [0, 1, 2, 3]
        )

        graph.value_info.extend([
            starts_tensor_info,
            ends_tensor_info,
            axes_tensor_info
        ])

    # -------- Insert Conv nodes for each slice --------
    for i in range(plan.n_o_h_seg):
        pad_top, pad_left, pad_bottom, pad_right = spec.pads

        # Only apply original padding at boundaries
        pad_top = pad_top if i == 0 else 0
        pad_bottom = pad_bottom if i == plan.n_o_h_seg - 1 else 0

        conv_node = helper.make_node(
            "Conv",
            inputs=[spec.in_name + '_slice_' + str(i), spec.k_name],
            outputs=[node.name + '_out_' + str(i)],
            kernel_shape=[spec.k_h, spec.k_w],
            strides=spec.strides,
            pads=[pad_top, pad_left, pad_bottom, pad_right]
        )
        graph.node.append(conv_node)

    # -------- Concatenate outputs back together --------
    concat_node = helper.make_node(
        "Concat",
        inputs=[node.name + '_out_' + str(i) for i in range(plan.n_o_h_seg)],
        outputs=[spec.out_name],
        axis=1   # NOTE: concatenating along channel axis
    )
    graph.node.append(concat_node)

    # -------- Add tensor shape metadata --------
    for i in range(plan.n_o_h_seg):
        starts = max(plan.o_h * spec.strides[0] * i - (spec.k_h - 1) // 2, 0)
        ends = min(plan.o_h * spec.strides[0] * (i + 1) + (spec.k_h - 1) // 2, spec.in_h)

        # Adjust last segment output height
        o_h_t = plan.o_h if i != plan.n_o_h_seg - 1 else spec.out_h - (i * plan.o_h)

        in_tensor_info = helper.make_tensor_value_info(
            spec.in_name + '_slice_' + str(i),
            TensorProto.FLOAT,
            [spec.batch, spec.in_channel, ends - starts, spec.in_w]
        )
        out_tensor_info = helper.make_tensor_value_info(
            node.name + '_out_' + str(i),
            TensorProto.FLOAT,
            [spec.batch, spec.out_channel, o_h_t, spec.out_w]
        )

        graph.value_info.extend([in_tensor_info, out_tensor_info])

    # -------- Remove original Conv node --------
    remove_nodes(graph, lambda n: n == node)

