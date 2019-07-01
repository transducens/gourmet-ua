import sys
IL_PREFIX="interleaved_"
for line in sys.stdin:
    line=line.rstrip("\n")
    toks=line.split(" ")
    il=[t for t in toks if t.startswith(IL_PREFIX)]
    il_clean=[ t[len(IL_PREFIX):] for t in il]
    print(" ".join(il_clean))
