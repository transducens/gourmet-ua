#! /bin/bash
set -euo pipefail

CHECKPOINTDIR="$1"
DEVSL="$2"
DATABIN="$3"
VALIDATESCRIPT="$4"
TRANSLATEARGS="$5"
GPUS="$6"

NAME="$7"

if [ $# -ge 8  ]; then
	TAGSMODE="$8"
else
	TAGSMODE=""
fi

SCORESFILE="scores$NAME"

rm -f $CHECKPOINTDIR/$SCORESFILE
for CP in $CHECKPOINTDIR/checkpoint*.pt ; do
  # Translate and evaluate
  SCORE=$( CUDA_VISIBLE_DEVICES=$GPUS fairseq-interactive $TRANSLATEARGS  --input $DEVSL --path $CP $DATABIN | tee $CP.output$NAME.raw  |  {  if [ "$TAGSMODE" != "tags"  ]; then  cat - | grep '^H-' | cut -f 3 ; else  cat - | grep '^TAGS:'| cut -f 2- -d ' ' ; fi  } | tee $CP.output$NAME | $VALIDATESCRIPT )
  echo "$CP $SCORE" >> $CHECKPOINTDIR/scores$NAME
done

#Find the best checkpoint and symlink
BESTMODEL=$( LC_ALL=C sort -nr -k2,2 -t ' ' < $CHECKPOINTDIR/scores$NAME | head -n 1 | cut -f 1 -d ' ' )
ln -s $(readlink -f $BESTMODEL) $CHECKPOINTDIR/checkpoint_best_metric$NAME.pt
