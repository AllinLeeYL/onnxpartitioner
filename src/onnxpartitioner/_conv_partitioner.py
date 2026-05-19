import numpy as np
import onnx_graphsurgeon as gs
from dataclasses import dataclass

from .common import *
from .graphsurgeon_utils import add_node, get_parents, get_parent_from_tensor, get_successors, remove_node


# def try_partition_conv(graph: gs.Graph, node: gs.Node, hardware: dict, direction: str, plan_func=None):
#     partition_plan = plan_func if plan_func != None else default_partition_plan
#     # Step 1: extract parameters
#     spec = conv_params(graph, node)

#     # Step 2: compute plan
#     plan = partition_plan(spec, hardware, direction)
#     if plan == None:
#         return False
#     print("Partition plan:", plan, "applied to", node.name)

#     # Step 3: apply transformation
#     last_node = apply_conv_partition(graph, node, spec, plan)
#     if not plan.concat_node:
#         if plan.n_out_channel != 0:
#             new_plan = ConvPartitionPlan(n_in_channel=plan.n_out_channel, in_channel_s=plan.out_channel_s, do_slice=False)
#             for sub_node in get_successors(last_node):
#                 print("Partition plan:", plan, "applied to", node.name)
#                 apply_conv_partition(graph, sub_node, spec, new_plan)
#         elif plan.vertical:
#             pass
#         else:
#             pass
#     elif not plan.sum_node:
#         pass

#     return True


@dataclass
class ConvPartitionPlan:
    n_in_channel: int = 0 # number of input channel segment
    in_channel_s: int = 0 # partitioned channel size
    n_out_channel: int = 0 # number of output channel segment
    out_channel_s: int = 0 # partitioned channel size
    n_o_seg: int = 0 # number of output height/weight segment
    o_hw: int = 0 # size of output height/weight segment
    vertical: bool = True # True: segment by height / False: by weight

    # advanced control. This may affect other nodes
    concat_node: bool = True # whether to add concat node after
    sum_node: bool = True # whether to add add node after
    do_slice: bool = True # whether do slice. If not, the input is from the last layer.


def conv_params(graph: gs.Graph, node: gs.Node):
    inp = node.inputs[0]
    outp = node.outputs[0]
    kernel = node.inputs[1]
    bias = node.inputs[2]

    return ConvSpec(
        in_h=inp.shape[2],
        in_w=inp.shape[3],
        in_name=inp.name,
        in_channel=inp.shape[1],

        out_h=outp.shape[2],
        out_w=outp.shape[3],
        out_name=outp.name,
        out_channel=outp.shape[1],

        k_h=kernel.shape[2],
        k_w=kernel.shape[3],
        k_name=kernel.name,
        b_name=bias.name,
        batch=outp.shape[0],
        strides=node.attrs["strides"],
        pads=node.attrs["pads"]
    )


def default_partition_plan(spec: ConvSpec, hardware, direction):
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


def apply_conv_partition(graph: gs.Graph, node: gs.Node, spec: ConvSpec, plan: ConvPartitionPlan):
    assert(node.op == "Conv")
    if plan.n_in_channel!=0:
        last_node = input_channel_partition(graph, node, spec, plan)
    elif plan.n_out_channel!=0:
        last_node = output_channel_partition(graph, node, spec, plan)
    elif plan.vertical:
        last_node = height_partition(graph, node, spec, plan)
    else:
        last_node = width_partition(graph, node, spec, plan)
    graph.cleanup().toposort()
    return last_node


def input_channel_partition(graph: gs.Graph, node: gs.Node, spec: ConvSpec, plan: ConvPartitionPlan):
    """
    input channel partition
    """
    zero_bias = gs.Constant(
        name=spec.b_name + '_zero',
        values=np.zeros(spec.out_channel, dtype=node.inputs[2].dtype)
    )
    sub_op_outs = []
    if not plan.do_slice:
        concat = get_parent_from_tensor(node.inputs[0])
        assert(concat.op == "Concat")
    for i in range(0, plan.n_in_channel):
        kernel = np.array(node.inputs[1].values, dtype=node.inputs[1].dtype)
        bias = node.inputs[2]
        starts = i*plan.in_channel_s
        ends = min(spec.in_channel, (i+1)*plan.in_channel_s)
        if plan.do_slice:
            # -------- Insert Slice nodes (split input feature map) --------
            sliced_tensor = gs.Variable(
                name=spec.in_name + '_slice_' + str(i) + '_for_' + node.name,
                dtype=node.outputs[0].dtype,
                shape=[spec.batch, ends-starts, spec.in_h, spec.in_w]
            )
            starts_tensor = gs.Constant(
                name=spec.in_name + '_slice_starts_' + str(i) + '_for_' + node.name,
                values=np.array([0, starts, 0, 0], dtype=np.int64)
            )
            ends_tensor = gs.Constant(
                name=spec.in_name + '_slice_ends_' + str(i) + '_for_' + node.name,
                values=np.array([spec.batch, ends, spec.in_h, spec.in_w], dtype=np.int64)
            )
            axes_tensor = gs.Constant(
                name=spec.in_name + '_slice_axes_' + str(i) + '_for_' + node.name,
                values=np.array([0, 1, 2, 3], dtype=np.int64)
            )

            inputs = [
                node.inputs[0],
                starts_tensor,
                ends_tensor,
                axes_tensor
            ]
            outputs = [sliced_tensor]
            graph.add_node(op="Slice", inputs=inputs, outputs=outputs)

        # sliced_kernel = kernel[:, starts:ends, :, :]
        else:
            sliced_tensor = concat.inputs[i]
        
        sliced_kernel = gs.Constant(
            name=spec.k_name + '_slice_' + str(i),
            values=kernel[:, starts:ends, :, :]
        )
        inputs = [
            sliced_tensor,
            sliced_kernel,
            bias if i==plan.n_in_channel-1 else zero_bias
        ]
        sub_op_out = gs.Variable(
            name=node.name + '_out_' + str(i),
            dtype=node.outputs[0].dtype,
            shape=[spec.batch, spec.out_channel, spec.out_h, spec.out_w]
        )
        sub_op_outs.append(sub_op_out)
        attrs = {"strides": spec.strides, "pads": spec.pads}
        graph.add_node(
            op="Conv", 
            name=node.name+'_ic_sub'+str(i), 
            inputs=inputs, 
            outputs=[sub_op_out], 
            attrs=attrs
        )
    if not plan.do_slice:
        graph.remove_node(concat)
    # -------- Sum outputs back together --------
    outputs = graph.add_node(op="Sum", name="sum_for_"+node.name, inputs=sub_op_outs, outputs=[node.outputs[0]])
    graph.remove_node(node)
    return get_parent_from_tensor(outputs[0])


def output_channel_partition(graph: gs.Graph, node:gs.Node, spec: ConvSpec, plan: ConvPartitionPlan):
    is_last = node.outputs == graph.outputs
    concat_inputs = []
    for i in range(0, plan.n_out_channel):
        kernel = np.array(node.inputs[1].values, dtype=node.inputs[1].dtype)
        bias = np.array(node.inputs[2].values, dtype=node.inputs[2].dtype)
        starts = i*plan.out_channel_s
        ends = min((i+1)*plan.out_channel_s, spec.out_channel)
        
        # ------ Conv node -----
        inputs = [
            node.inputs[0],
            gs.Constant(name=spec.k_name + '_slice_' + str(i), values=kernel[starts:ends, :, :, :]),
            gs.Constant(name=spec.b_name + '_slice_' + str(i), values=bias[starts:ends])
        ]
        sliced_out_tensor = gs.Variable(name=node.name + '_out_' + str(i), dtype=node.outputs[0].dtype, shape=[spec.batch, ends-starts, spec.out_h, spec.out_w])
        concat_inputs.append(sliced_out_tensor)
        attrs = {
            "strides": spec.strides,
            "pads": spec.pads
        }
        graph.add_node(
            op="Conv", 
            name=node.name + '_oc_sub' + str(i),
            inputs=inputs,
            outputs=[sliced_out_tensor],
            attrs=attrs
        )
    outputs = graph.add_node(op="Concat", inputs=concat_inputs, outputs=[node.outputs[0]], attrs={"axis": 1})
    graph.remove_node(node)
    return get_parent_from_tensor(outputs[0])



def height_partition(graph: gs.Graph, node: gs.Node, spec: ConvSpec, plan: ConvPartitionPlan):
    sub_op_outs = []
    for i in range(0, plan.n_o_seg):
        pad_top, pad_left, pad_bottom, pad_right = spec.pads
        stride = spec.strides[0]

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

        # -------- Insert Slice nodes (split input feature map) --------
        sliced_tensor = gs.Variable(
            name=spec.in_name + '_slice_' + str(i) + '_for_' + node.name,
            dtype=node.outputs[0].dtype,
            shape=[spec.batch, spec.in_channel, ends-starts, spec.in_w]
        )
        inputs = [
            node.inputs[0],
            gs.Constant(
                name=spec.in_name + '_slice_starts_' + str(i) + '_for_' + node.name,
                values=np.array([0, 0, starts, 0], dtype=np.int64)
            ),
            gs.Constant(
                name=spec.in_name + '_slice_ends_' + str(i) + '_for_' + node.name,
                values=np.array([spec.batch, spec.in_channel, ends, spec.in_w], dtype=np.int64)
            ),
            gs.Constant(
                name=spec.in_name + '_slice_axes_' + str(i) + '_for_' + node.name,
                values=np.array([0, 1, 2, 3], dtype=np.int64)
            )
        ]

        graph.add_node(
            op="Slice",
            inputs=inputs,
            outputs=[sliced_tensor]
        )
        # -------- Insert Conv nodes for each slice --------
        out_h = plan.o_hw if (i+1)*plan.o_hw <= spec.out_h else spec.out_h - i*plan.o_hw
        sub_op_out = gs.Variable(
            name=node.name + '_out_' + str(i),
            dtype=node.outputs[0].dtype,
            shape=[spec.batch, spec.out_channel, out_h, spec.out_w]
        )
        sub_op_outs.append(sub_op_out)
        graph.add_node(
            op="Conv",
            name=node.name+"_oh_sub"+str(i),
            inputs=[sliced_tensor, node.inputs[1], node.inputs[2]],
            outputs=[sub_op_out],
            attrs={
                "strides": spec.strides,
                "pads": [pad_top, pad_left, pad_bottom, pad_right]
            }
        )

    # -------- Concatenate outputs back together --------
    outputs = graph.add_node(
        op="Concat",
        inputs=sub_op_outs,
        outputs=[node.outputs[0]],
        attrs={"axis": 2}
    )
    graph.remove_node(node)
    return get_parent_from_tensor(outputs[0])


def width_partition(graph: gs.Graph, node: gs.Node, spec: ConvSpec, plan: ConvPartitionPlan):
    sub_op_outs = []
    for i in range(0, plan.n_o_seg):
        # params
        pad_top, pad_left, pad_bottom, pad_right = spec.pads
        stride = spec.strides[1] # TODO: check which one is vertical?
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

        # -------- Insert Slice nodes (split input feature map) --------
        sliced_tensor = gs.Variable(
            name=spec.in_name + '_slice_' + str(i) + '_for_' + node.name,
            dtype=node.outputs[0].dtype,
            shape=[spec.batch, spec.in_channel, spec.in_h, ends-starts]
        )
        inputs = [
            node.inputs[0],
            gs.Constant(
                name=spec.in_name + '_slice_starts_' + str(i) + '_for_' + node.name,
                values=np.array([0, 0, 0, starts], dtype=np.int64)
            ),
            gs.Constant(
                name=spec.in_name + '_slice_ends_' + str(i) + '_for_' + node.name,
                values=np.array([spec.batch, spec.in_channel, spec.in_h, ends], dtype=np.int64)
            ),
            gs.Constant(
                name=spec.in_name + '_slice_axes_' + str(i) + '_for_' + node.name,
                values=np.array([0, 1, 2, 3], dtype=np.int64)
            )
        ]
        graph.add_node(
            op="Slice",
            inputs=inputs,
            outputs=[sliced_tensor]
        )
        # -------- Insert Conv nodes for each slice --------
        out_w = plan.o_hw if (i+1)*plan.o_hw <= spec.out_w else spec.out_w - i*plan.o_hw
        sub_op_out = gs.Variable(
            name=node.name + '_out_' + str(i),
            dtype=node.outputs[0].dtype,
            shape=[spec.batch, spec.out_channel, spec.out_h, out_w]
        )
        sub_op_outs.append(sub_op_out)
        graph.add_node(
            op="Conv",
            name=node.name+"_sub"+str(i),
            inputs=[sliced_tensor, node.inputs[1], node.inputs[2]],
            outputs=[sub_op_out],
            attrs={
                "strides": spec.strides,
                "pads": [pad_top, pad_left, pad_bottom, pad_right]
            }
        )

    # -------- Concatenate outputs back together --------
    outputs = graph.add_node(
        op="Concat",
        inputs=sub_op_outs,
        outputs=[node.outputs[0]],
        attrs={"axis": 3}
    )
    graph.remove_node(node)
    return get_parent_from_tensor(outputs[0])
    
