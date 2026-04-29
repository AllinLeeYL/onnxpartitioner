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
                        help='path to model2.onnx file.')
    args = parser.parse_args()
    return args


def run_model(path, input_data):
    sess = ort.InferenceSession(path)
    return sess.run(None, input_data)


if __name__ == '__main__':
    args = parse_argument()
    for i in range(0, args.num):
        inp = np.random.randn(1,24, 256,256).astype(np.float32)
        out1 = run_model(args.model1, {"input": inp})
        out2 = run_model(args.model2, {"input": inp})
        if not np.allclose(out1[0], out2[0], atol=1e-5):
            print(colored("Consistency check failed", "red"))
    print(colored("Consistency check passed", "green"))