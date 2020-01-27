#!/bin/bash

# to be called, obviously, as 
# $ randomness-exp.sh $EXP_NAME $EXP_ROOT $DATA_ROOT

EXP_NAME=$1
EXP_ROOT=$2
DATA_ROOT=$3

echo ""
echo "*** BLOCK EVALUATION"
export CUDA_VISIBLE_DEVICES=$(~/util/sge/get-free-gpus.py --max=1);
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
