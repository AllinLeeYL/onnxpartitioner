
import argparse
import torch
import torch.nn as nn
import random
import subprocess
import os
from tqdm import tqdm
from termcolor import colored
import warnings
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=20, 
                        help='path to model2.onnx file.')
    parser.add_argument('--repeat', type=int, default=3,
                        help='repeat number of checks for each model.')
    parser.add_argument('--use_cuda', action='store_true',
                        help="Enable CUDA execution")
    parser.add_argument('--multiprocess', action='store_true',
                        help="Accelerating with multiprocess")
    parser.add_argument('--rtol', type=float, default=1e-4, 
                        help='The relative tolerance parameter.')
    parser.add_argument('--atol', type=float, default=1e-5, 
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


def build_process_params(args):
    print('building models:')
    params = []
    for i in tqdm(range(args.num)):
        # model parameters
        in_channels=random.randint(1, 256)
        img_h = random.randint(16, 512)
        img_w = random.randint(16, 512)

        # limit parameters
        in_channel_limit = random.randint(1, 256)
        out_channel_limit = random.randint(1, 256)
        in_pixel_limit = random.randint(256, 8192)
        out_pixel_limit = random.randint(256, 8192)

        # out = model(dummy_input)
        onnx_path = f"test/model_{in_channels}_in_{img_h}x{img_w}.onnx"
        partitioned_path = f"test/model_{in_channels}_in_{img_h}x{img_w}_partitioned.onnx"
        
        # partition
        partition_cmds = ["python3", 
                "partitioner.py", 
                onnx_path, 
                "--in_channel", str(in_channel_limit),
                "--in_pixel", str(in_pixel_limit),
                "--out_channel", str(out_channel_limit),
                "--out_pixel", str(out_pixel_limit)]

        # check
        check_cmds = ["python3", "check.py", "--rtol", str(args.rtol), "--atol", str(args.atol), onnx_path, partitioned_path]
        if args.use_cuda:
            check_cmds.extend(["--use_cuda"])
        
        params.append([in_channels, img_h, img_w, partition_cmds, check_cmds, onnx_path, partitioned_path])

    return params


def run(params):
    warnings.filterwarnings("ignore", category=FutureWarning)
    in_channels, img_h, img_w, partition_cmds, check_cmds, onnx_path, partitioned_path = params
    # dummy input and model
    k_h = random.choice([1, 3, 5, 7, 9, 11, 13])
    k_w = random.choice([1, 3, 5, 7, 9, 11, 13])
    dummy_input = torch.randn(1, in_channels, img_h, img_w).to(device)
    model = ConvOnlyNet(in_channels=in_channels, 
                        num_layers=random.randint(1, 1),
                        k=(k_h, k_w) ).to(device)
    # test model
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        verbose=False
    )
    p = subprocess.run(partition_cmds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0 and b"The kernel size is too big. Cannot partition." in p.stderr:
        return

    if p.returncode != 0 and b"paddings are too big." in p.stderr:
        return
    p = subprocess.run(check_cmds, stdout=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RuntimeError("Conv2d partitioner consistency check failed. The last partition command was:\n" + " ".join(partition_cmds) + "\nAnd the last checking command was:\n" + " ".join(check_cmds))
    try:
        os.remove(onnx_path)
        os.remove(onnx_path+".data")
        os.remove(partitioned_path)
    except:
        pass


def single_process_main(args):
    for i in tqdm(range(args.num)):
        # model parameters
        in_channels=random.randint(1, 256)
        img_h = random.randint(16, 512)
        img_w = random.randint(16, 512)
        k_h = random.choice([1, 3, 5, 7, 9, 11, 13])
        k_w = random.choice([1, 3, 5, 7, 9, 11, 13])

        # limit parameters
        in_channel_limit = random.randint(1, 256)
        out_channel_limit = random.randint(1, 256)
        in_pixel_limit = random.randint(256, 8192)
        out_pixel_limit = random.randint(256, 8192)

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
        partition_cmds = ["python3", 
                "partitioner.py", 
                onnx_path, 
                "--in_channel", str(in_channel_limit),
                "--in_pixel", str(in_pixel_limit),
                "--out_channel", str(out_channel_limit),
                "--out_pixel", str(out_pixel_limit)]
        p = subprocess.run(partition_cmds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if p.returncode != 0 and b"The kernel size is too big. Cannot partition." in p.stderr:
            continue

        if p.returncode != 0 and b"paddings are too big." in p.stderr:
            continue

        # check
        check_cmds = ["python3", "check.py", "--rtol", str(args.rtol), "--atol", str(args.atol), onnx_path, partitioned_path]
        if args.use_cuda:
            check_cmds.extend(["--use_cuda"])
        p = subprocess.run(check_cmds, stdout=subprocess.DEVNULL)
        if p.returncode != 0:
            raise RuntimeError("Conv2d partitioner consistency check failed. The last partition command was:\n" + " ".join(partition_cmds) + "\nAnd the last checking command was:\n" + " ".join(check_cmds))
        try:
            os.remove(onnx_path)
            os.remove(onnx_path+".data")
            os.remove(partitioned_path)
        except:
            pass
    print(colored("Conv2d partitioner consistency check passed", "green"))


if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    # -----------------------------
    # Create output folder
    # -----------------------------
    os.makedirs("test", exist_ok=True)
    
    args = parse_argument()

    if args.multiprocess:
        mp.set_start_method("spawn", force=True)
        params = build_process_params(args)
        with mp.Pool(3) as p:
            results = list(
            tqdm(
                p.imap(run, params),
                total=len(params)
            )
        )
    else:
        single_process_main(args)