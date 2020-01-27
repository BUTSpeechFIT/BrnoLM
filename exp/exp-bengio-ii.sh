#!/bin/bash

# to be called, obviously, as 
# $ randomness-exp.sh $SEED $EXP_ROOT $EXP_NAME $IVEC_DIM $IVEC_EXTRACTOR $SIZ

SEED=$1
EXP_ROOT=$2
EXP_NAME=$3
SIZE=$4
DROPOUT=$5
HIST_LEN=$6
IVEC_DIM=$7
IVEC_EXTRACTOR=$8
DATA_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos

# python build_bengio_ivec_input.py \
#     --wordlist=$DATA_ROOT/wordlist.txt \
#     --save=$EXP_ROOT/$EXP_NAME.init.lm \
#     --unk="<unk>" \
#     --nhid=$SIZE \
#     --emsize=$SIZE \
#     --dropout=$DROPOUT \
#     --ivec-dim=$IVEC_DIM \
#     --hist-len=$HIST_LEN || exit 1

sleep $[ ( $RANDOM % 180)  + 1 ]s
export CUDA_VISIBLE_DEVICES=$(~/util/sge/get-free-gpus.py --max=1);

# python train-ff-multifile-ivec-oracle.py \
#     --train-list=$DATA_ROOT/train-list.txt \
#     --valid-list=$DATA_ROOT/valid-list.txt \
#     --test-list=$DATA_ROOT/test-list.txt \
#     --seed=$SEED \
#     --concat-articles \
#     --keep-shuffling \
#     --cuda \
#     --lr=1 \
#     --batch_size=20 \
#     --min-batch-size=10 \
#     --target-seq-len=30 \
#     --ivec-extractor=$IVEC_EXTRACTOR \
#     --load=$EXP_ROOT/$EXP_NAME.init.lm \
#     --save=$EXP_ROOT/$EXP_NAME.lm \
#     --epochs=35 || exit 1


for TESTLIST in train-list.txt valid-list.txt test-list.txt
do
    python eval-ff-multifile-ivecs.py \
        --file-list=$DATA_ROOT/$TESTLIST \
        --cuda \
        --batch_size=20 \
        --target-seq-len=10 \
        --ivec-extractor=$IVEC_EXTRACTOR \
        --load=$EXP_ROOT/$EXP_NAME.lm
done
