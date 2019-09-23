#! /bin/bash
MYFULLPATH=$(readlink -f $0)
CURDIR=$(dirname $MYFULLPATH)


WDIR="$1"
GPU="$2"
SL="$3"
TL="$4"

#Re-translate training corpus forcing factors
#CUDA_VISIBLE_DEVICES=$GPUID python3 $CURDIR/interactive.py --user-dir /neural/gourmet-ua/fairseqmodules --task translation_tlfactors --input $WDIR/corpus/train.clean-bpe.$SL --force-surface-forms $WDIR/corpus/train.clean-bpe.nointerl.$TL --print-factors --path $WDIR/model/checkpoints/checkpoint_best_tag_perplexity_frozen.pt $WDIR/model/data-bin > $WDIR/corpus/train.clean-bpe.nointerl.$TL.re-translated.forcedsf.raw

#Extract tags
grep '^TAGS:' $WDIR/corpus/train.clean-bpe.nointerl.$TL.re-translated.forcedsf.raw | cut -f 2- -d ':' | sed 's:^[ ]*::' > $WDIR/corpus/train.clean-bpe.nointerl.$TL.re-translated.forcedsf.tags

#Replace tags in training corpuys
python3 $CURDIR/replace-interleaving-tags.py $WDIR/corpus/train.clean-bpe.nointerl.$TL.re-translated.forcedsf.tags < $WDIR/corpus/train.clean-bpe.$TL > $WDIR/corpus/train.clean-bpe.jackniffedtags.$TL
cp $WDIR/corpus/train.clean-bpe.$SL $WDIR/corpus/train.clean-bpe.jackniffedtags.$SL

#Create new data-bin
python3 $CURDIR/fairseq_preprocess_factors.py --joined-dictionary  -s $SL -t $TL  --trainpref $WDIR/corpus/train.clean-bpe.jackniffedtags --validpref $WDIR/corpus/dev.bpe --destdir $WDIR/model/data-bin-jackniffing --workers 16 --additional_decoder_tl --srcdict $WDIR/model/data-bin/dict.${SL}.txt --tgtfactorsdict $WDIR/model/data-bin/dict.${TL}factors.txt
