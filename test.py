import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_idx', type=int)

args = parser.parse_args()
run = args.run_idx

print(run)