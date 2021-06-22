import argparse, sys
sys.path.append('/scratch/oem214/vanilla-rtrl/')
from wrappers import compare_analyzed_checkpoints

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')

args = parser.parse_args()

analysis_job_name = 'analyze_{}'.format(args.name)

compare_analyzed_checkpoints(analysis_job_name)