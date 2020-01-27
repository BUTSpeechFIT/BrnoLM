#!/bin/bash

# to be called, obviously, as 
# $ eval-eval-batchsize.sh $MODEL $MIN_BATCH_SIZE $MAX_BATCH_SIZE

MODEL=$1
MIN_BATCH_SIZE=$2
MAX_BATCH_SIZE=$3
EXP_ROOT=/mnt/matylda5/ibenes/projects/santosh-lm/lms/wt2/lstm-mid
DATA_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos

for BSZ in $(seq $MIN_BATCH_SIZE $MAX_BATCH_SIZE);
do
    python eval.py --data=$DATA_ROOT/pythlm-symlinks --cuda --batch_size=$BSZ --bptt=35 --load=$MODEL
done
