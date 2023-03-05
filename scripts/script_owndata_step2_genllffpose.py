'''
Takes in the scenedir and register the camera poses
'''
from colmaps import gen_poses
import sys
import argparse

parser = argparse.ArgumentParser(description='colmap')
parser.add_argument('--scenedir', type=str, required=True)
parser.add_argument('--share_intrin', action='store_true')
args = parser.parse_args()

scenedir = args.scenedir
share_intrin = args.share_intrin
factors = 1
use_lowres = False
match_type = 'exhaustive_matcher'  # exhaustive_matcher or sequential_matcher

# =======================================================

if __name__ == '__main__':
    gen_poses(scenedir, match_type, factors, usedown=use_lowres, share_intrin=share_intrin)
