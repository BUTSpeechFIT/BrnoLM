#!/bin/bash

SEED=$1
EXP_ROOT=$2
EXP_NAME=$3
SIZE=$4
DROPOUT=$5
HIST_LEN=$6
DATA_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos
BIN_DIR=/homes/kazi/ibenes/PhD/pyth-lm/balls/scripts

echo "******************"
echo "*** Building model"
echo "******************"
python $BIN_DIR/model_building/build_bengio.py \
    --wordlist=$DATA_ROOT/wordlist.txt \
    --save=$EXP_ROOT/$EXP_NAME.init.lm \
    --unk="<unk>" \
    --nhid=$SIZE \
    --emsize=$SIZE \
    --dropout=$DROPOUT \
    --hist-len=$HIST_LEN || exit 1

sleep $[ ( $RANDOM % 120 )  + 1 ]s

echo "******************"
echo "*** Training model"
echo "******************"
export CUDA_VISIBLE_DEVICES=$(~/util/sge/get-free-gpus.py --max=1);
python $BIN_DIR/train/train-multifile.py \
    --train-list=$DATA_ROOT/train-list.txt \
    --valid-list=$DATA_ROOT/valid-list.txt \
    --seed=$SEED \
    --concat-articles \
    --keep-shuffling \
    --cuda \
    --lr=1 \
    --batch-size=20 \
    --min-batch-size=10 \
    --target-seq-len=30 \
    --load=$EXP_ROOT/$EXP_NAME.init.lm \
    --save=$EXP_ROOT/$EXP_NAME.lm \
    --epochs=35 || exit 1


echo "******************"
echo "*** Testing model"
echo "******************"
for TESTLIST in train-list.txt valid-list.txt test-list.txt
do
    python $BIN_DIR/eval/eval-multifile.py \
        --file-list=$DATA_ROOT/$TESTLIST \
        --cuda \
        --batch-size=20 \
        --target-seq-len=10 \
        --load=$EXP_ROOT/$EXP_NAME.lm
done

echo "******************"
echo "*** Domain adaptation as baseline"
echo "******************"
for TESTLIST in train-list.txt valid-list.txt test-list.txt
do
    python $BIN_DIR/eval/eval-noivecs-domain-adaptation.py \
        --file-list=$DATA_ROOT/$TESTLIST \
        --cuda \
        --batch-size=20 \
        --domain-portion=0.25 \
        --target-seq-len=10 \
        --load=$EXP_ROOT/$EXP_NAME.lm
done
