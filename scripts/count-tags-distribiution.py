import sys
import argparse
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser(description="Counts translations associated with each tag")
parser.add_argument('out_file', type=str, help="Pickled output file")
args = parser.parse_args()

TAGPREFIX="interleaved_"
BPESUFFIX="@@"

d = {}

for line in sys.stdin:
    toks=line.rstrip("\n").split()
    lastTag=None
    for t in toks:
        if t.startswith(TAGPREFIX):
            lastTag=t
        else:
            queryTag=lastTag
            if t.endswith(BPESUFFIX):
                queryTag=lastTag+BPESUFFIX
            if queryTag not in d:
                d[queryTag]=defaultdict(int)
            d[queryTag][t]+=1

pickle.dump( d, open( args.out_file, "wb" ) )
