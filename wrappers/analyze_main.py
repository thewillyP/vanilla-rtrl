import argparse, os, pickle
from wrappers.analyze_training_run import analyze_training_run

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

saved_run_name = args.name

analyze_training_run(saved_run_name)