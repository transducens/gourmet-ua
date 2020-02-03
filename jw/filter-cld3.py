import sys
import cld3
import argparse

oparser = argparse.ArgumentParser(description="Tool that reads a list of pairs of segments from the standard input in TSV "
        "format, this is: a pair of segments per line, segments separated by tabs. The script will discard any pair for "
        "which the langauge detector CLD3 identifies that one of the segments is not in the language expected, or if the "
        "identification is not reliable.")
oparser.add_argument("--lang1", help="2-character code of the language to be detected by CLD3 in the first field of the input (for some languages, it could be a 3-character code; see CLD3 documentation for more information)", dest="lang1", required=True)
oparser.add_argument("--lang2", help="2-character code of the language to be detected by CLD3 in the second fields of the input (for some languages, it could be a 3-character code; see CLD3 documentation for more information)", dest="lang2", required=True)
oparser.add_argument("-v", "--verbose", help="If this option is enabled, the script outputs to the error output messages describing why a pair of segments has been discarded", action="store_true", default=False)

options = oparser.parse_args()

for line in sys.stdin:
    line=line.rstrip("\n")
    lines=line.split("\t")
    line1=lines[0].strip()
    lang1=cld3.get_language(line1)
    line2=lines[1].strip()
    lang2=cld3.get_language(line2)
    
    if len(line1) == 0 or len(line2) == 0:
        continue

    if lang1.is_reliable != True:
        if options.verbose:
            sys.stderr.write("LANG1 not reliable: "+str(lang1)+"\t"+line1+"\n")
    elif lang1.language!=options.lang1:
        if options.verbose:
            sys.stderr.write("LANG1 not "+options.lang1+": "+str(lang1)+"\t"+line1+"\n")
    elif lang2.is_reliable != True:
        if options.verbose:
            sys.stderr.write("LANG2 not reliable: "+str(lang2)+"\t"+line2+"\n")
    elif lang2.language!=options.lang2:
        if options.verbose:
            sys.stderr.write("LANG2 not "+options.lang2+": "+str(lang2)+"\t"+line2+"\n")
    else:
        print(line.strip())

