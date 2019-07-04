import sys
IL_PREFIX="interleaved_"
for line in sys.stdin:
    line=line.rstrip("\n")
    toks=line.split(" ")
    il=[t for t in toks if not t.startswith(IL_PREFIX)]
    print(" ".join(il))
