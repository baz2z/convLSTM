import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--run_idx', type=int)

args = parser.parse_args()
run = args.run_idx

a = torch.ones([10, 32, 32])
l, w, h = a.shape
print(l)
print(run)