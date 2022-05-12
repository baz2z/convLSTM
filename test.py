import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_idx', type=int)

args = parser.parse_args()
run = args.run_idx

print(run)