#!/usr/bin/env bash
EXP_DIR=$1
EXP_NAME=$2
DATA_ROOT=$3
IVEC_EXTRACTOR=$4

python balls/scripts/model-building/build-shallow-nn.py \
    --wordlist=$DATA_ROOT/wordlist.txt \
    --unk="<unk>" \
    --emsize=20 \
    --nhid=20 \
    --save=$EXP_DIR/$EXP_NAME.init.lm || exit 1

# 1) train the FN-LM 
python balls/scripts/train/train-multifile.py \
    --train-list=$DATA_ROOT/valid-list.txt \
    --valid-list=$DATA_ROOT/test-list.txt \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.init.lm \
    --save=$EXP_DIR/$EXP_NAME.lm \
    --epochs=1 || exit 1

python balls/scripts/eval/eval-multifile.py \
    --file-list=$DATA_ROOT/valid-list.txt \
    --concat-articles \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.lm  || exit 1

python balls/scripts/eval/eval-noivecs-domain-adaptation.py \
    --file-list=$DATA_ROOT/valid-list.txt \
    --concat-articles \
    --domain-portion=0.25 \
    --cuda \
    --load=$EXP_DIR/$EXP_NAME.lm  || exit 1
