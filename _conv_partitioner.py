import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, TensorProto
# import onnxruntime as ort
import numpy as np

from common import *
from dataclasses import dataclass


def conv_exceed_hardware_limit(graph, node, hardware):
    buffers = calculate_conv_buf(graph, node)
    for buf_name, buf, in buffers.items():
        if buf > hardware[buf_name]:
            return True
    
    return False


def partition_conv(graph, node, hardware):
    # Step 1: extract
    spec = conv_params(graph, node)

    # Step 2: compute plan
    plan = compute_partition_plan(spec, hardware)

    # Step 3: apply transformation
    graph = apply_conv_partition(graph, node, spec, plan)

    print("Partition plan:", plan, "applied to", node.name)

    return graph


@dataclass
class _PartitionPlan:
    n_o_seg: int # number of output height/weight segment
    o_hw: int # size of output height/weight segment
    vertical: bool # True: segment by height / False: by weight


def conv_params(graph, node):
    in_name = node.input[0]
    out_name = node.output[0]
    kernel_name = node.input[1]
    bias_name = node.input[2]
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
        b_name=bias_name,
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
        if spec.in_h > spec.in_w:
            in_len = spec.in_w
            out_len = spec.out_w
            out_row = spec.out_h
            k = spec.k_h
            stride = spec.strides[0]
        else:
            in_len = spec.in_h
            out_len = spec.out_h
            out_row = spec.out_w
            k = spec.k_w
            stride = spec.strides[1]
        # -------- Partition along height/weight dimension --------

        # Compute max input tile height/weight that fits in input buffer
        i_h = max_multiplier_within_limit(
            base=in_len,
            limit=hardware['input_buffer'].pixel_s - in_len * (k - 1)
        )

        # Corresponding output height/weight from input tiling
        o_h_by_input = i_h // stride

        # Compute max output tile height/weight that fits in output buffer
        o_h = max_multiplier_within_limit(
            base=out_len,
            limit=hardware['output_buffer'].pixel_s - out_len * (k - 1)
        )

        # Final tile height/weight constrained by both input and output limits
        o_h = min(o_h, o_h_by_input)

        # if kernel is too big
        if o_h <= 0:
            raise RuntimeError("The kernel size is too big. Cannot partition.")

        # Number of segments needed to cover full output
        q, r = divmod(out_row, o_h)
        n_seg = q if r == 0 else q + 1
        if spec.in_h > spec.in_w:
            return _PartitionPlan(o_hw=o_h, n_o_seg=n_seg, vertical=True)
        else:
            return _PartitionPlan(o_hw=o_h, n_o_seg=n_seg, vertical=False)


def apply_conv_partition(graph, node, spec: ConvSpec, plan: _PartitionPlan):
    for i in range(0, plan.n_o_seg):
        # params
        pad_top, pad_left, pad_bottom, pad_right = spec.pads
        pad_start, pad_end = (pad_top, pad_bottom) if plan.vertical else (pad_left, pad_right)
        pad_other_start, pad_other_end = (pad_left, pad_right) if plan.vertical else (pad_top, pad_bottom)
        k = spec.k_h if plan.vertical else spec.k_w
        in_len = spec.in_h if plan.vertical else spec.in_w
        in_other_len = spec.in_w if plan.vertical else spec.in_h
        stride = spec.strides[0] if plan.vertical else spec.strides[1] # TODO: check which one is vertical?
        # -------- Insert Slice nodes (split input feature map) --------
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
        # starts = plan.o_h * spec.strides[0] * i - spec.k_h//2
        starts = plan.o_hw * stride * i - pad_start
        starts = max(starts, 0)
        # ends = plan.o_h * spec.strides[0] * (i + 1) + spec.k_h//2
        ends = plan.o_hw * stride * (i + 1) - pad_start + k//2
        ends = min(ends, in_len)

        if ends < 0 or starts > in_len:
            raise RuntimeError("paddings are too big. kernel size=" + str(k) + ", padding=" + str(spec.pads))

        # Add tensor metadata for slice parameters
        starts_tensor = helper.make_tensor(
            name=spec.in_name + '_slice_starts_' + str(i),
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[0, 0, starts, 0] if plan.vertical else [0, 0, 0, starts]
        )
        ends_tensor = helper.make_tensor(
            name=spec.in_name + '_slice_ends_' + str(i),
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[spec.batch, spec.in_channel, ends, spec.in_w] if plan.vertical else [spec.batch, spec.in_channel, spec.in_h, ends]
        )
        axes_tensor = helper.make_tensor(
            name=spec.in_name + '_slice_axes_' + str(i),
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[0, 1, 2, 3]
        )
        graph.initializer.extend([
            starts_tensor,
            ends_tensor,
            axes_tensor
        ])

        # -------- Insert Conv nodes for each slice --------
        # Only apply original padding at boundaries
        pad_start = pad_start if i == 0 else 0
        pad_end = 0 if ends <= in_len else min(pad_end, ends - in_len)

        conv_node = helper.make_node(
            "Conv",
            name=node.name+"_"+str(i),
            inputs=[spec.in_name + '_slice_' + str(i), spec.k_name, spec.b_name],
            outputs=[node.name + '_out_' + str(i)],
            kernel_shape=[spec.k_h, spec.k_w],
            strides=spec.strides,
            pads=[pad_start, pad_other_start, pad_end, pad_other_end] if plan.vertical else [pad_other_start, pad_start, pad_other_end, pad_end]
        )
        graph.node.append(conv_node)

    # -------- Concatenate outputs back together --------
    concat_node = helper.make_node(
        "Concat",
        inputs=[node.name + '_out_' + str(i) for i in range(plan.n_o_seg)],
        outputs=[spec.out_name],
        axis=2 if plan.vertical else 3   # NOTE: concatenating along height/weight
    )
    graph.node.append(concat_node)

    # -------- Remove original Conv node --------
    remove_nodes(graph, lambda n: n == node)
    return graph

