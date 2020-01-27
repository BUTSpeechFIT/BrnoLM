#!/bin/bash

# to be called, obviously, as 
# $ wt2-midsize-exp.sh $SEED $EXP_NAME

SEED=$1
EXP_NAME=$2
EXP_ROOT=/mnt/matylda5/ibenes/projects/santosh-lm/lms/wt2/lstm-mid
DATA_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos

python build_lstm.py --seed=$SEED --wordlist=$DATA_ROOT/wordlist.txt --save=$EXP_ROOT/$EXP_NAME-init.lm --unk="<unk>" --nhid=600 --emsize=600 --dropout=0.5

python train-multifile.py --train-list=$DATA_ROOT/train-list.txt --valid-list=$DATA_ROOT/valid-list.txt --test-list=$DATA_ROOT/test-list.txt --seed=$SEED --concat-articles --keep-shuffling --cuda --lr=20 --batch_size=20 --min-batch-size=10 --bptt=35 --load=$EXP_ROOT/$EXP_NAME-init.lm --save=$EXP_ROOT/$EXP_NAME.lm --epochs=20

echo ""
echo "*** BLOCK EVALUATION"
python eval.py --data=$DATA_ROOT/pythlm-symlinks --cuda --batch_size=20 --bptt=35 --load=$EXP_ROOT/$EXP_NAME.lm

echo ""
echo "*** MULTIFILE EVALUATION -- TRAIN"
python eval-multifile.py --file-list=$DATA_ROOT/train-list.txt --batch_size=20 --bptt=35 --cuda --load=$EXP_ROOT/$EXP_NAME.lm

echo ""
echo "*** MULTIFILE EVALUATION -- VALID"
python eval-multifile.py --file-list=$DATA_ROOT/valid-list.txt --batch_size=20 --bptt=35 --cuda --load=$EXP_ROOT/$EXP_NAME.lm

echo ""
echo "*** MULTIFILE EVALUATION -- TEST"
python eval-multifile.py --file-list=$DATA_ROOT/test-list.txt --batch_size=20 --bptt=35 --cuda --load=$EXP_ROOT/$EXP_NAME.lm
