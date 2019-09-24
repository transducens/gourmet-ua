import sys
import argparse


parser = argparse.ArgumentParser(description="Extract n-grams from a tokenized file")
parser.add_argument('n', type=int, default=4)
args = parser.parse_args()

for line in sys.stdin:
    line=line.rstrip("\n")
    toks=line.split(" ")
    ngrams = zip(*[toks[i:] for i in range(args.n)])
    for ngram in ngrams:
        print(" ".join(ngram))
