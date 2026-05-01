import onnx
import onnxruntime as ort
import numpy as np
import argparse
import torch
from termcolor import colored

sess_options = ort.SessionOptions()
# Set graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1', type=str,  
                        help='path to model1.onnx file.')
    parser.add_argument('model2', type=str,  
                        help='path to model2.onnx file.')
    parser.add_argument('--use_cuda', action='store_true',
                        help="Enable CUDA execution")
    parser.add_argument('--num', type=int, default=20, 
                        help='number of tests.')
    parser.add_argument('--rtol', type=float, default=1e-2, 
                        help='The relative tolerance parameter.')
    parser.add_argument('--atol', type=float, default=1e-3, 
                        help='The absolute tolerance parameter.')
    args = parser.parse_args()
    return args


def tensor_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        # dynamic dims may have dim_param instead of dim_value
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(1)  # fallback for dynamic dimensions
    return shape


def get_session(path, cuda=True):
    providers = ["CUDAExecutionProvider"] if cuda else []
    providers += ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, 
                                sess_options=sess_options,
                                providers=providers)
    return sess


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parse_argument()
    sess1 = get_session(args.model1, args.use_cuda)
    sess2 = get_session(args.model2, args.use_cuda)

    for i in range(0, args.num):
        model = onnx.load(args.model1)
        input_tensor = model.graph.input[0]
        inp = np.random.randn(*tensor_shape(input_tensor)).astype(np.float32)
        # print(inp)
        out1 = sess1.run(None, {"input": inp})
        out1 = np.array(out1)
        out2 = sess2.run(None, {"input": inp})
        out2 = np.array(out2)
        if not np.allclose(out1, out2, rtol=args.rtol, atol=args.atol):
            max_diff = np.max(np.abs(out1 - out2))
            max_diff = round(max_diff, 6)
            raise RuntimeError("Consistency check failed! " + "The maximum diff is " + str(max_diff))
            exit(1)
    print(colored("Consistency check passed", "green"))