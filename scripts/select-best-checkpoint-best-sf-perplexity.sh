#!/bin/bash

MODELDIR="$1"

BEST_UPDATE=$(grep 'valid on' $MODELDIR/../train.log | cut -f 7,9 -d '|' | grep ' loss_a' | LC_ALL=C sort -n -k 5,5  | head -n 1 | cut -f 3 -d  ' ')
echo "Copying checkpoint $BEST_UPDATE to checkpoint_last with comand: cp $MODELDIR/checkpoint_*_$BEST_UPDATE.pt $MODELDIR/checkpoint_last.pt"
cp $MODELDIR/checkpoint_*_$BEST_UPDATE.pt $MODELDIR/checkpoint_last.pt
cp $MODELDIR/checkpoint_*_$BEST_UPDATE.pt $MODELDIR/checkpoint_best_sf_perplexity_frozen.pt
