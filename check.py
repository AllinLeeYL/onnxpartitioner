import onnxruntime as ort
import numpy as np
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1', type=str,  
                        help='path to model1.onnx file.')
    parser.add_argument('model2', type=str,  
                        help='path to model2.onnx file.')
    args = parser.parse_args()
    return args


def run_model(path, input_data):
    sess = ort.InferenceSession(path)
    return sess.run(None, input_data)


if __name__ == '__main__':
    args = parse_argument()
    out1 = run_model(args.model1, {"input": np.random.randn(1,3,32,32).astype(np.float32)})
    out2 = run_model(args.model2, {"input": np.random.randn(1,3,32,32).astype(np.float32)})
    print(np.allclose(out1[0], out2[0], atol=1e-5))