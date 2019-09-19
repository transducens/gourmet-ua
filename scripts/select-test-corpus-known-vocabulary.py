import sys
import argparse

parser = argparse.ArgumentParser("Prints the line numbers of a SL test corpus (read from stdin) all whose words are observed at training time ")
parser.add_argument("--print-unknown",action='store_true',help="Prints only the lines that contain unknown words.")
parser.add_argument("vocabulary",help="Plain text file with vocabulary, one word per line")
args = parser.parse_args()

v=set()
with open(args.vocabulary) as v_f:
    for line in v_f:
        line=line.rstrip("\n")
        v.add(line)


ln=0
for l in sys.stdin:
    ln+=1
    l=l.rstrip("\n")
    words=l.split()
    if all(w in v for w in words):
        if not args.print_unknown:
            print(ln)
    else:
        if args.print_unknown:
            print(ln)
