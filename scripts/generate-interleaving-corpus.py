import sys
import argparse

PREFIX="interleaved_"
BPEMARK="@@"

parser = argparse.ArgumentParser("Generates an interleaving corpus file. Puts the morphosyntactic information before each bped word, with the prefix '{}'".format(PREFIX))
#parser.add_argument("bped_corpus",help="tokenized and BPEd corpus file")
parser.add_argument("factored_corpus",help="tokenized and factored corpus")
args = parser.parse_args()


with open(args.factored_corpus) as f_factored:
	for x, y in zip(sys.stdin, f_factored):
		toks_bped=x.strip().split(" ")
		toks_factored=y.strip().split(" ")
		starttokens=[t for t in toks_bped if not t.endswith(BPEMARK)]
		if len(starttokens) != len(toks_factored):
			print("Token length mismatch", file=sys.stderr)
			exit(1)
		out_toks=[]
		positionFactored=0
		for i,bpedtok in enumerate(toks_bped):
			prevtoken=None
			if i > 0:
				prevtoken=toks_bped[i-1]
			if prevtoken == None or  not prevtoken.endswith(BPEMARK):
				out_toks.append(PREFIX+toks_factored[ positionFactored])
				positionFactored+=1
			out_toks.append(bpedtok)
		print(" ".join(out_toks))
