
import argparse
import torch
import torch.nn as nn
import random
import subprocess
import os
from tqdm import tqdm
from termcolor import colored
import warnings

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=20, 
                        help='path to model2.onnx file.')
    parser.add_argument('--use_cuda', action='store_true',
                        help="Enable CUDA execution")
    parser.add_argument('--rtol', type=float, default=1e-2, 
                        help='The relative tolerance parameter.')
    parser.add_argument('--atol', type=float, default=1e-3, 
                        help='The absolute tolerance parameter.')
    args = parser.parse_args()
    return args


class ConvOnlyNet(nn.Module):
    def __init__(self, in_channels, num_layers, k):
        super().__init__()
        layers = []
        c = in_channels

        for i in range(num_layers):
            out_c = random.choice([i for i in range(16, 128)])

            layers.append(nn.Conv2d(c, out_c, kernel_size=k, padding=random.choice([i for i in range(0, max(k))])))
            # layers.append(nn.ReLU())

            c = out_c

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=FutureWarning)
    # -----------------------------
    # Create output folder
    # -----------------------------
    os.makedirs("test", exist_ok=True)

    args = parse_argument()
    for i in tqdm(range(args.num)):
        # parameters
        in_channels=random.randint(1, 64)
        img_h = random.randint(16, 512)
        img_w = random.randint(16, 512)
        k_h = random.choice([1, 3, 5, 7, 9, 11, 13])
        k_w = random.choice([1, 3, 5, 7, 9, 11, 13])

        # dummy input and model
        dummy_input = torch.randn(1, in_channels, img_h, img_w).to(device)
        model = ConvOnlyNet(in_channels=in_channels, 
                            num_layers=random.randint(1, 1),
                            k=(k_h, k_w) ).to(device)

        # test model
        model.eval()
        # out = model(dummy_input)
        onnx_path = f"test/model_{in_channels}_in_{img_h}x{img_w}.onnx"
        partitioned_path = f"test/model_{in_channels}_in_{img_h}x{img_w}_partitioned.onnx"

        # export model
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            verbose=False
        )
        
        # partition
        p = subprocess.run(["python3", "partitioner.py", onnx_path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if p.returncode != 0 and b"The kernel size is too big. Cannot partition." in p.stderr:
            continue

        # check
        cmds = ["python3", "check.py", "--rtol", str(args.rtol), "--atol", str(args.atol), onnx_path, partitioned_path]
        if args.use_cuda:
            cmds.extend("--use_cuda")
        p = subprocess.run(cmds, stdout=subprocess.DEVNULL)
        if p.returncode != 0:
            raise RuntimeError("Conv2d partitioner consistency check failed")
        try:
            os.remove(onnx_path)
            os.remove(onnx_path+".data")
            os.remove(partitioned_path)
        except:
            pass
    print(colored("Conv2d partitioner consistency check passed", "green"))