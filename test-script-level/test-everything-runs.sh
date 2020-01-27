#!/usr/bin/env bash
EXP_DIR=$1

WT_ROOT=/mnt/matylda5/ibenes/text-data/wikitext-2-explicit-eos/
IVEC_EXTRACTOR=/mnt/matylda5/ibenes/projects/santosh-lm/smm-models/wt2-model/extractor-k50

# note that only validation part of WT-2 is used for training to speed the test up
TEST_DIR="${BASH_SOURCE%/*}"

# 1) build a standard LSTM
$TEST_DIR/test-lstm-runs.sh $EXP_DIR lsmt $WT_ROOT || exit 1

# 2) build a SMM-LSTM
# commented out as we cannost fit output-enhaced SMM-LSTM  into the decoder framework
# $TEST_DIR/test-smm-lstm-runs.sh $EXP_DIR smm-lstm $WT_ROOT $IVEC_EXTRACTOR || exit 1

# 3) build iFN-LM
$TEST_DIR/test-fn-runs.sh $EXP_DIR ifn $WT_ROOT  || exit 1

# 4) build iFN-LM
$TEST_DIR/test-ifn-runs.sh $EXP_DIR ifn $WT_ROOT $IVEC_EXTRACTOR || exit 1
