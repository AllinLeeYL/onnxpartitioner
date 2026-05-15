import torch
import onnx
from onnx import shape_inference, numpy_helper, helper, TensorProto
# import onnxruntime as ort
import numpy as np
from dataclasses import dataclass

from .common import *



# def conv_exceed_hardware_limit(graph, node, hardware):
#     buffers = calculate_conv_buf(graph, node)
#     for buf_name, buf, in buffers.items():
#         if buf > hardware[buf_name]:
#             return True
    
#     return False


def try_partition_conv(graph, node, hardware, direction: str, plan_func=None):
    partition_plan = plan_func if plan_func != None else compute_partition_plan
    # Step 1: extract
    spec = conv_params(graph, node)

    # Step 2: compute plan
    plan = partition_plan(spec, hardware, direction)
    if plan == None:
        return False
    print("Partition plan:", plan, "applied to", node.name)

    # Step 3: apply transformation
    graph = apply_conv_partition(graph, node, spec, plan)

    return True


@dataclass
class ConvPartitionPlan:
    n_in_channel: int = 0 # number of input channel segment
    in_channel_s: int = 0 # partitioned channel size
    n_out_channel: int = 0 # number of output channel segment
    out_channel_s: int = 0 # partitioned channel size
    n_o_seg: int = 0 # number of output height/weight segment
    o_hw: int = 0 # size of output height/weight segment
    vertical: bool = True # True: segment by height / False: by weight
    concat_node: bool = True # whether to add concat node after
    sum_node: bool = True # whether to add add node after


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


# def calculate_conv_buf(graph, node):
#     spec = conv_params(graph, node)

#     return {'input_buffer': Buffer(spec.in_channel, spec.in_h * spec.in_w),
#             'output_buffer': Buffer(spec.out_channel, spec.out_h * spec.out_w)}


def compute_partition_plan(spec: ConvSpec, hardware, direction):
    buffers = {'input_buffer': Buffer(spec.in_channel, spec.in_h * spec.in_w),
               'output_buffer': Buffer(spec.out_channel, spec.out_h * spec.out_w)}
    do_partition = False
    for buf_name, buf, in buffers.items():
        if buf > hardware[buf_name]:
            do_partition = True
            break
    if not do_partition:
        return None
    
    if spec.out_channel > hardware['output_buffer'].channel_s:
        q, r = divmod(spec.out_channel, hardware['output_buffer'].channel_s)
        n_seg = q if r == 0 else q + 1
        return ConvPartitionPlan(n_out_channel=n_seg, out_channel_s=hardware['output_buffer'].channel_s)
    elif spec.in_channel > hardware['input_buffer'].channel_s:
        q, r = divmod(spec.in_channel, hardware['input_buffer'].channel_s)
        n_seg = q if r == 0 else q + 1
        return ConvPartitionPlan(n_in_channel=n_seg, in_channel_s=hardware['input_buffer'].channel_s)
    else:
        is_vertical = True if direction=='vertical' else False if direction=='horizontal' else spec.in_h >= spec.in_w
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
            limit=hardware['output_buffer'].pixel_s# - out_len * (k - 1)
        )

        # Final tile height/weight constrained by both input and output limits
        o_h = min(o_h, o_h_by_input)

        # if kernel is too big
        if o_h <= 0:
            raise RuntimeError("The kernel size is too big. Cannot partition.")

        # Number of segments needed to cover full output
        q, r = divmod(out_row, o_h)
        n_seg = q if r == 0 else q + 1
        
        return ConvPartitionPlan(n_in_channel=0, in_channel_s=0,
                              n_out_channel=0, out_channel_s=0,
                              o_hw=o_h, n_o_seg=n_seg, vertical=is_vertical)


def apply_conv_partition(graph, node, spec: ConvSpec, plan: ConvPartitionPlan):
    if plan.n_in_channel!=0:
        input_channel_partition(graph, node, spec, plan)
    elif plan.n_out_channel!=0:
        output_channel_partition(graph, node, spec, plan)
    elif plan.vertical:
        height_partition(graph, node, spec, plan)
    else:
        width_partition(graph, node, spec, plan)
    # -------- Remove original Conv node --------
    remove_nodes(graph, lambda n: n == node)
    return graph


def input_channel_partition(graph, node, spec: ConvSpec, plan: ConvPartitionPlan):
    """
    input channel partition
    """
    init_map = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    for i in range(0, plan.n_in_channel):
        kernel = init_map[spec.k_name]
        starts = i*plan.in_channel_s
        ends = min(spec.in_channel, (i+1)*plan.in_channel_s)
        # -------- Insert Slice nodes (split input feature map) --------
        sliced_input_name = spec.in_name + '_slice_' + str(i) + '_for_' + node.name
        sliced_input_starts_name = spec.in_name + '_slice_starts_' + str(i) + '_for_' + node.name
        sliced_input_ends_name = spec.in_name + '_slice_ends_' + str(i) + '_for_' + node.name
        sliced_input_axes_name = spec.in_name + '_slice_axes_' + str(i) + '_for_' + node.name
        slice_node = helper.make_node(
            "Slice",
            inputs=[
                spec.in_name,
                sliced_input_starts_name,
                sliced_input_ends_name,
                sliced_input_axes_name
            ],
            outputs=[sliced_input_name],
        )
        graph.node.append(slice_node)

        # Add slice parameters
        starts_tensor = helper.make_tensor(
            name=sliced_input_starts_name,
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[0, starts, 0, 0]
        )
        ends_tensor = helper.make_tensor(
            name=sliced_input_ends_name,
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[spec.batch, ends, spec.in_h, spec.in_w]
        )
        axes_tensor = helper.make_tensor(
            name=sliced_input_axes_name,
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
        zero_bias_name = spec.b_name + '_zero'
        conv_node = helper.make_node(
            'Conv',
            name=node.name+'_ic_sub'+str(i),
            inputs=[sliced_input_name, 
                    spec.k_name + '_slice_' + str(i), 
                    spec.b_name if i == 0 else zero_bias_name],
            outputs=[node.name + '_out_' + str(i)],
            kernel_shape=[spec.k_h, spec.k_w],
            strides=spec.strides,
            pads=spec.pads
        )
        graph.node.append(conv_node)

        # -------- Insert Conv kernels  --------
        kernel_tensor = helper.make_tensor(
            name=spec.k_name + '_slice_' + str(i),
            data_type=TensorProto.FLOAT,
            dims=[spec.out_channel, ends-starts, spec.k_h, spec.k_w],
            vals=kernel[:, starts:ends, :, :]
        )
        graph.initializer.extend([
            kernel_tensor
        ])
        # Create a ValueInfoProto
        value_info = helper.make_tensor_value_info(
            name=spec.k_name + '_slice_' + str(i),
            elem_type=TensorProto.FLOAT,
            shape=[spec.out_channel, ends-starts, spec.k_h, spec.k_w]
        )
        graph.value_info.append(value_info)
    # -------- Insert zero bias  --------
    zero_bias_tensor = helper.make_tensor(
        name=zero_bias_name,
        data_type=TensorProto.FLOAT,
        dims=[spec.out_channel],
        vals=np.zeros(spec.out_channel)
    )
    graph.initializer.extend([
        zero_bias_tensor
    ])
    # -------- Sum outputs back together --------
    add_node = helper.make_node(
        'Sum',
        inputs=[node.name + '_out_' + str(i) for i in range(plan.n_in_channel)],
        outputs=[spec.out_name],
        # axis=1   # NOTE: concatenating along channel
    )
    graph.node.append(add_node)

    # -------- Remove unused Conv parameters --------
    remove_names = {spec.k_name}
    keep = [
        init for init in graph.initializer
        if init.name not in remove_names
    ]
    del graph.initializer[:]
    graph.initializer.extend(keep)


def output_channel_partition(graph, node, spec: ConvSpec, plan: ConvPartitionPlan):
    init_map = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    for i in range(0, plan.n_out_channel):
        kernel = init_map[spec.k_name]
        bias = init_map[spec.b_name]
        starts = i*plan.out_channel_s
        ends = min((i+1)*plan.out_channel_s, spec.out_channel)
        # ------ Conv node -----
        conv_node = helper.make_node(
            'Conv',
            name=node.name + '_oc_sub' + str(i),
            inputs=[spec.in_name,
                    spec.k_name + '_slice_' + str(i),
                    spec.b_name + '_slice_' + str(i)],
            outputs=[node.name + '_out_' + str(i)],
            kernel_shape=[spec.k_h, spec.k_w],
            strides=spec.strides,
            pads=spec.pads
        )
        graph.node.append(conv_node)
        # ----- Kernel and bias -----
        kernel_tensor = helper.make_tensor(
            name=spec.k_name + '_slice_' + str(i),
            data_type=TensorProto.FLOAT,
            dims=[ends-starts, spec.in_channel, spec.k_h, spec.k_w],
            vals=kernel[starts:ends, :, :, :]
        )
        bias_tensor = helper.make_tensor(
            name=spec.b_name + '_slice_' + str(i),
            data_type=TensorProto.FLOAT,
            dims=[ends-starts],
            vals=bias[starts:ends]
        )
        graph.initializer.extend([kernel_tensor, bias_tensor])
        # Create ValueInfo
        kernel_info = helper.make_tensor_value_info(
            name=spec.k_name + '_slice_' + str(i),
            elem_type=TensorProto.FLOAT,
            shape=[ends-starts, spec.in_channel, spec.k_h, spec.k_w]
        )
        bias_info = helper.make_tensor_value_info(
            name=spec.b_name + '_slice_' + str(i),
            elem_type=TensorProto.FLOAT,
            shape=[ends-starts]
        )
        graph.value_info.extend([kernel_info, bias_info])

    if plan.concat_node:
        concat_node = helper.make_node(
            'Concat',
            inputs=[node.name + '_out_' + str(i) for i in range(plan.n_out_channel)],
            outputs=[spec.out_name],
            axis=1
        )
        graph.node.append(concat_node)
        remove_names = {spec.k_name, spec.b_name}
    else:
        remove_names = {spec.k_name, spec.b_name, spec.out_name}

    # -------- Remove unused Conv parameters --------
    keep = [
        init for init in graph.initializer
        if init.name not in remove_names
    ]
    del graph.initializer[:]
    graph.initializer.extend(keep)


def height_partition(graph, node, spec: ConvSpec, plan: ConvPartitionPlan):
    init_map = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    for i in range(0, plan.n_o_seg):
        pad_top, pad_left, pad_bottom, pad_right = spec.pads
        stride = spec.strides[0]
        # -------- Insert Slice nodes (split input feature map) --------
        sliced_input_name = spec.in_name + '_slice_' + str(i) + '_for_' + node.name
        sliced_input_starts_name = spec.in_name + '_slice_starts_' + str(i) + '_for_' + node.name
        sliced_input_ends_name = spec.in_name + '_slice_ends_' + str(i) + '_for_' + node.name
        sliced_input_axes_name = spec.in_name + '_slice_axes_' + str(i) + '_for_' + node.name
        slice_node = helper.make_node(
            "Slice",
            inputs=[
                spec.in_name,
                sliced_input_starts_name,
                sliced_input_ends_name,
                sliced_input_axes_name
            ],
            outputs=[sliced_input_name],
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

        if ends <= 0 or starts >= spec.in_h:
            raise RuntimeError("paddings are too big. kernel size=" + str(spec.k_h) + ", padding=" + str(spec.pads))
        
        assert(starts != ends)

        # Add tensor metadata for slice parameters
        starts_tensor = helper.make_tensor(
            name=sliced_input_starts_name,
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[0, 0, starts, 0]
        )
        ends_tensor = helper.make_tensor(
            name=sliced_input_ends_name,
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[spec.batch, spec.in_channel, ends, spec.in_w]
        )
        axes_tensor = helper.make_tensor(
            name=sliced_input_axes_name,
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
            name=node.name+"_oh_sub"+str(i),
            inputs=[sliced_input_name, spec.k_name, spec.b_name],
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


def width_partition(graph, node, spec: ConvSpec, plan: ConvPartitionPlan):
    init_map = {
        init.name: numpy_helper.to_array(init)
        for init in graph.initializer
    }
    for i in range(0, plan.n_o_seg):
        # params
        pad_top, pad_left, pad_bottom, pad_right = spec.pads
        stride = spec.strides[1] # TODO: check which one is vertical?
        # -------- Insert Slice nodes (split input feature map) --------
        sliced_input_name = spec.in_name + '_slice_' + str(i) + '_for_' + node.name
        sliced_input_starts_name = spec.in_name + '_slice_starts_' + str(i) + '_for_' + node.name
        sliced_input_ends_name = spec.in_name + '_slice_ends_' + str(i) + '_for_' + node.name
        sliced_input_axes_name = spec.in_name + '_slice_axes_' + str(i) + '_for_' + node.name
        slice_node = helper.make_node(
            "Slice",
            inputs=[
                spec.in_name,
                sliced_input_starts_name,
                sliced_input_ends_name,
                sliced_input_axes_name
            ],
            outputs=[sliced_input_name],
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

        if ends <= 0 or starts >= spec.in_w:
            raise RuntimeError("paddings are too big. kernel size=" + str(spec.k_w) + ", padding=" + str(spec.pads))
        
        assert(starts != ends)

        # Add tensor metadata for slice parameters
        starts_tensor = helper.make_tensor(
            name=sliced_input_starts_name,
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[0, 0, 0, starts]
        )
        ends_tensor = helper.make_tensor(
            name=sliced_input_ends_name,
            data_type=TensorProto.INT32,
            dims=[4],
            vals=[spec.batch, spec.in_channel, spec.in_h, ends]
        )
        axes_tensor = helper.make_tensor(
            name=sliced_input_axes_name,
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
            name=node.name+"_sub"+str(i),
            inputs=[sliced_input_name, spec.k_name, spec.b_name],
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
    

