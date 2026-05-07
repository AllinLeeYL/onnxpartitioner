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


def partition_conv(graph, node, hardware, direction: str):
    # Step 1: extract
    spec = conv_params(graph, node)

    # Step 2: compute plan
    plan = compute_partition_plan(spec, hardware, direction)

    # Step 3: apply transformation
    graph = apply_conv_partition(graph, node, spec, plan)

    print("Partition plan:", plan, "applied to", node.name)

    return graph


@dataclass
class _PartitionPlan:
    n_in_channel: int = 0 # number of input channel segment
    in_channel_s: int = 0 # partitioned channel size
    n_out_channel: int = 0 # number of output channel segment
    out_channel_s: int = 0 # partitioned channel size
    n_o_seg: int = 0 # number of output height/weight segment
    o_hw: int = 0 # size of output height/weight segment
    vertical: bool = True# True: segment by height / False: by weight


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


def compute_partition_plan(spec: ConvSpec, hardware, direction):
    if spec.out_channel > hardware['output_buffer'].channel_s:
        q, r = divmod(spec.out_channel, hardware['output_buffer'].channel_s)
        n_seg = q if r == 0 else q + 1
        return _PartitionPlan(n_out_channel=n_seg, out_channel_s=hardware['output_buffer'].channel_s)
    elif spec.in_channel > hardware['input_buffer'].channel_s:
        q, r = divmod(spec.in_channel, hardware['input_buffer'].channel_s)
        n_seg = q if r == 0 else q + 1
        return _PartitionPlan(n_in_channel=n_seg, in_channel_s=hardware['input_buffer'].channel_s)
    else:
        is_vertical = True if direction=='vertical' else False if direction=='horizontal' else spec.in_h > spec.in_w
        if is_vertical:
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
        
        return _PartitionPlan(n_in_channel=0, in_channel_s=0,
                              n_out_channel=0, out_channel_s=0,
                              o_hw=o_h, n_o_seg=n_seg, vertical=is_vertical)


def apply_conv_partition(graph, node, spec: ConvSpec, plan: _PartitionPlan):
    init_map = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    # -------------- input channel partition --------------
    if plan.n_in_channel!=0:
        for i in range(0, plan.n_in_channel):
            channel_s = plan.in_channel_s if i != plan.n_in_channel-1 else spec.in_channel - i*plan.in_channel_s

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

            # Add slice parameters
            starts_tensor = helper.make_tensor(
                name=spec.in_name + '_slice_starts_' + str(i),
                data_type=TensorProto.INT32,
                dims=[4],
                vals=[0, 0, 0, 0]
            )
            ends_tensor = helper.make_tensor(
                name=spec.in_name + '_slice_ends_' + str(i),
                data_type=TensorProto.INT32,
                dims=[4],
                vals=[spec.batch, channel_s, spec.in_h, spec.in_w]
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
            conv_node = helper.make_node(
                "Conv",
                name=node.name+"_"+str(i),
                inputs=[spec.in_name + '_slice_' + str(i), 
                        spec.k_name + '_slice_' + str(i), 
                        spec.b_name],
                outputs=[node.name + '_out_' + str(i)],
                kernel_shape=[spec.k_h, spec.k_w],
                strides=spec.strides,
                pads=spec.pads
            )
            graph.node.append(conv_node)

            # -------- Insert Conv kernels --------
            kernel = init_map[spec.k_name]
            # print(kernel.shape)
            starts = i*plan.in_channel_s
            ends = min(spec.in_channel, (i+1)*plan.in_channel_s)
            kernel_tensor = helper.make_tensor(
                name=spec.k_name + '_slice_' + str(i),
                data_type=TensorProto.FLOAT,
                dims=[spec.out_channel, ends-starts, spec.k_h, spec.k_w],
                vals=kernel[:, starts:ends, :, :]
            )
            print(kernel_tensor.dims, kernel_tensor.name)
            graph.initializer.extend([
                kernel_tensor
            ])
            

        # -------- Concatenate outputs back together --------
        concat_node = helper.make_node(
            "Concat",
            inputs=[node.name + '_out_' + str(i) for i in range(plan.n_in_channel)],
            outputs=[spec.out_name],
            axis=1   # NOTE: concatenating along channel
        )
        graph.node.append(concat_node)

    # -------------- output channel partition --------------
    elif plan.n_out_channel!=0:
        pass

    # -------------- height channel partition --------------
    elif plan.vertical:
        for i in range(0, plan.n_o_seg):
            pad_top, pad_left, pad_bottom, pad_right = spec.pads
            stride = spec.strides[0]
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
            # Note the computation order here matters
            starts = plan.o_hw * stride * i - pad_top
            ends = (plan.o_hw * (i + 1) - 1) * stride - pad_top + spec.k_h
            # update paddings
            pad_top = 0 if starts >= 0 else abs(starts)
            pad_bottom = 0 if ends <= spec.in_h else min(pad_bottom, ends - spec.in_h)
            # clamp the starts and ends
            starts = max(starts, 0)
            ends = min(ends, spec.in_h)

            if ends < 0 or starts > spec.in_h:
                raise RuntimeError("paddings are too big. kernel size=" + str(spec.k_h) + ", padding=" + str(spec.pads))

            # Add tensor metadata for slice parameters
            starts_tensor = helper.make_tensor(
                name=spec.in_name + '_slice_starts_' + str(i),
                data_type=TensorProto.INT32,
                dims=[4],
                vals=[0, 0, starts, 0]
            )
            ends_tensor = helper.make_tensor(
                name=spec.in_name + '_slice_ends_' + str(i),
                data_type=TensorProto.INT32,
                dims=[4],
                vals=[spec.batch, spec.in_channel, ends, spec.in_w]
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
            conv_node = helper.make_node(
                "Conv",
                name=node.name+"_"+str(i),
                inputs=[spec.in_name + '_slice_' + str(i), spec.k_name, spec.b_name],
                outputs=[node.name + '_out_' + str(i)],
                kernel_shape=[spec.k_h, spec.k_w],
                strides=spec.strides,
                pads=[pad_top, pad_left, pad_bottom, pad_right]
            )
            graph.node.append(conv_node)

        # -------- Concatenate outputs back together --------
        concat_node = helper.make_node(
            "Concat",
            inputs=[node.name + '_out_' + str(i) for i in range(plan.n_o_seg)],
            outputs=[spec.out_name],
            axis=2   # NOTE: concatenating along height
        )
        graph.node.append(concat_node)

    # -------------- weight channel partition --------------
    else:
        for i in range(0, plan.n_o_seg):
            # params
            pad_top, pad_left, pad_bottom, pad_right = spec.pads
            stride = spec.strides[1] # TODO: check which one is vertical?
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
            # Note the computation order here matters
            starts = plan.o_hw * stride * i - pad_left
            ends = (plan.o_hw * (i + 1) - 1) * stride - pad_left + spec.k_w
            # update paddings
            pad_left = 0 if starts >= 0 else abs(starts)
            pad_right = 0 if ends <= spec.in_w else min(pad_right, ends - spec.in_w)
            # clamp the starts and ends
            starts = max(starts, 0)
            ends = min(ends, spec.in_w)

            if ends < 0 or starts > spec.in_w:
                raise RuntimeError("paddings are too big. kernel size=" + str(spec.k_w) + ", padding=" + str(spec.pads))

            # Add tensor metadata for slice parameters
            starts_tensor = helper.make_tensor(
                name=spec.in_name + '_slice_starts_' + str(i),
                data_type=TensorProto.INT32,
                dims=[4],
                vals=[0, 0, 0, starts]
            )
            ends_tensor = helper.make_tensor(
                name=spec.in_name + '_slice_ends_' + str(i),
                data_type=TensorProto.INT32,
                dims=[4],
                vals=[spec.batch, spec.in_channel, spec.in_h, ends]
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
            conv_node = helper.make_node(
                "Conv",
                name=node.name+"_"+str(i),
                inputs=[spec.in_name + '_slice_' + str(i), spec.k_name, spec.b_name],
                outputs=[node.name + '_out_' + str(i)],
                kernel_shape=[spec.k_h, spec.k_w],
                strides=spec.strides,
                pads=[pad_top, pad_left, pad_bottom, pad_right]
            )
            graph.node.append(conv_node)

        # -------- Concatenate outputs back together --------
        concat_node = helper.make_node(
            "Concat",
            inputs=[node.name + '_out_' + str(i) for i in range(plan.n_o_seg)],
            outputs=[spec.out_name],
            axis=3   # NOTE: concatenating along weight
        )
        graph.node.append(concat_node)

    # -------- Remove original Conv node --------
    remove_nodes(graph, lambda n: n == node)
    return graph
    

