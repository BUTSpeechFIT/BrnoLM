#!/bin/bash
#
#$ -S /bin/bash
#$ -q long.q
#$ -l ram_free=4G,mem_free=4G,gpu=1

cd /homes/kazi/ibenes/PhD/pyth-lm
$CMD
