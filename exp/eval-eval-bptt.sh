#!/bin/bash

# to be called, obviously, as 
# $ eval-eval-bptt.sh $MODEL $MIN_BPTT $MAX_BPTT

MODEL=$1
MIN_BPTT=$2
MAX_BPTT=$3
EXP_ROOT=/mnt/matylda5/ibenes/projects/santosh-lm/lms/wt2/lstm-mid
DATA_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos

for BPTT in $(seq $MIN_BPTT $MAX_BPTT);
do
    python eval.py --data=$DATA_ROOT/pythlm-symlinks --cuda --batch_size=10 --bptt=$BPTT --load=$MODEL
    python eval-multifile.py --file-list=$DATA_ROOT/train-list.txt --cuda --batch_size=10 --bptt=$BPTT --load=$MODEL
    python eval-multifile.py --file-list=$DATA_ROOT/valid-list.txt --cuda --batch_size=10 --bptt=$BPTT --load=$MODEL
    python eval-multifile.py --file-list=$DATA_ROOT/test-list.txt --cuda --batch_size=10 --bptt=$BPTT --load=$MODEL
done
