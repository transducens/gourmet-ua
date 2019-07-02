import sys
IL_PREFIX="interleaved_"
MULTI_SUFFIX="@@"
for line in sys.stdin:
    line=line.rstrip("\n")
    toks=line.split(" ")
    out=[]
    lastTag=None
    for t in toks:
        if t.startswith(IL_PREFIX):
            lastTag=t
        else:
            if t.endswith(MULTI_SUFFIX):
                out.append(lastTag+MULTI_SUFFIX)
            else:
                out.append(lastTag)
    print(" ".join(out))
