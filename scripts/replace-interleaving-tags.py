import sys
import argparse
import warnings


PREFIX="interleaved_"
BPEMARK="@@"

parser = argparse.ArgumentParser("Generates a new interleaved corpys by replacing the tags with the ones provided")
parser.add_argument("new_tags",help="tokenized and factored corpus")
args = parser.parse_args()

with open(args.new_tags) as new_tags_f:
    for line,line_tags in zip(sys.stdin,new_tags_f):
        line=line.rstrip("\n")
        line_tags=line_tags.rstrip("\n")
        toks=line.split(" ")
        new_tags=[ t for t in line_tags.split(" ") if not t.endswith(BPEMARK)  ]
        replace=True
        if len( [tok for tok in toks if tok.startswith(PREFIX)]  ) > len(new_tags):
            #print(toks,file=sys.stderr)
            #print(new_tags,file=sys.stderr)
            warnings.warn("Too few new tags")
            replace=False
        if len( [tok for tok in toks if tok.startswith(PREFIX)]  ) > len(new_tags):
            warnings.warn("Too many new tags")
            replace=False
        if replace:
            for i,tok in enumerate(toks):
                if tok.startswith(PREFIX):
                    toks[i]=new_tags[0]
                    new_tags=new_tags[1:]
        print(" ".join(toks))
        

