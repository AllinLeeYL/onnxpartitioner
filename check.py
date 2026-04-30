import onnx
import onnxruntime as ort
import numpy as np
import argparse
from termcolor import colored

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1', type=str,  
                        help='path to model1.onnx file.')
    parser.add_argument('model2', type=str,  
                        help='path to model2.onnx file.')
    parser.add_argument('--num', type=int, default=20, 
                        help='number of tests.')
    parser.add_argument('--rtol', type=float, default=1e-4, 
                        help='The relative tolerance parameter.')
    parser.add_argument('--atol', type=float, default=1e-5, 
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


def run_model(path, input_data):
    sess = ort.InferenceSession(path)
    return sess.run(None, input_data)


if __name__ == '__main__':
    args = parse_argument()

    for i in range(0, args.num):
        model = onnx.load(args.model1)
        input_tensor = model.graph.input[0]
        inp = np.random.randn(*tensor_shape(input_tensor)).astype(np.float32)
        # print(inp)
        out1 = run_model(args.model1, {"input": inp})
        out2 = run_model(args.model2, {"input": inp})
        if not np.allclose(out1[0], out2[0], rtol=args.rtol, atol=args.atol):
            print(colored("Consistency check failed", "red"))
            exit(1)
    print(colored("Consistency check passed", "green"))