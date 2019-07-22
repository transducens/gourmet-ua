#!/bin/bash

MODELDIR="$1"

BEST_UPDATE=$(grep 'valid on' $MODELDIR/train.log | cut -f 7,11 -d '|' | grep ' loss_b' | LC_ALL=C sort -n -k 5,5  | head -n 1 | cut -f 3 -d  ' ')
echo "Copying checkpoint $BEST_UPDATED to checkpoint_last"
cp $MODELDIR/checkpoints/checkpoint_*_$BEST_UPDATE.pt $MODELDIR/checkpoints/checkpoint_last.pt
