#!/usr/bin/env bash

NBEST_DIR=$1 # where are outputs of lattice-to-nbests
PICKS_FILE=$2 # the output of rescoring-combine-scores.py
O_DIR=$3 # where to put the selected latts

mkdir -p $O_DIR

while read -r line
do
    segment=$(echo $LINE | cut -d' ' -f 1)
    best_trans=$(echo $LINE | cut -d' ' -f 2)

    cp $NBEST_DIR/$segment-$best_trans $O_DIR/$segment
done
