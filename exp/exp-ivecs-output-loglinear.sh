#!/bin/bash

# to be called, obviously, as 
# $ randomness-exp.sh $SEED $EXP_ROOT $EXP_NAME $IVEC_DIM $IVEC_EXTRACTOR

SEED=$1
EXP_ROOT=$2
EXP_NAME=$3
IVEC_DIM=$4
IVEC_EXTRACTOR=$5
IVEC_DROPOUT=0.0
IVEC_NB_ITERS=10
DATA_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos

python build_output_enhanced_lstm.py --wordlist=$DATA_ROOT/wordlist.txt --save=$EXP_ROOT/$EXP_NAME.init.lm --ivec-size=$IVEC_DIM --unk="<unk>" --nhid=600 --emsize=600 --dropout=0.5 --dropout-ivec=$IVEC_DROPOUT

sleep $[ ( $RANDOM % 120 )  + 1 ]s

export CUDA_VISIBLE_DEVICES=$(~/util/sge/get-free-gpus.py --max=1);
python train-multifile-ivecs.py --train-list=$DATA_ROOT/train-list.txt --valid-list=$DATA_ROOT/valid-list.txt --test-list=$DATA_ROOT/test-list.txt --ivec-extractor=$IVEC_EXTRACTOR --ivec-nb-iters=$IVEC_NB_ITERS --seed=$SEED --concat-articles --cuda --lr=20 --batch_size=20 --min-batch-size=10 --bptt=35 --load=$EXP_ROOT/$EXP_NAME.init.lm --save=$EXP_ROOT/$EXP_NAME.lm --epochs=35

echo ""
echo "*** TRAIN"
python eval-multifile-ivecs.py --file-list=$DATA_ROOT/train-list.txt --ivec-extractor=$IVEC_EXTRACTOR --ivec-nb-iters=$IVEC_NB_ITERS --cuda --batch_size=20 --bptt=35 --load=$EXP_ROOT/$EXP_NAME.lm

echo ""
echo "*** VALID"
python eval-multifile-ivecs.py --file-list=$DATA_ROOT/valid-list.txt --ivec-extractor=$IVEC_EXTRACTOR --ivec-nb-iters=$IVEC_NB_ITERS --cuda --batch_size=20 --bptt=35 --load=$EXP_ROOT/$EXP_NAME.lm

echo ""
echo "*** TEST"
python eval-multifile-ivecs.py --file-list=$DATA_ROOT/test-list.txt --ivec-extractor=$IVEC_EXTRACTOR --ivec-nb-iters=$IVEC_NB_ITERS --cuda --batch_size=20 --bptt=35 --load=$EXP_ROOT/$EXP_NAME.lm
