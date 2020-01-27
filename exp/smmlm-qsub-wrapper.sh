#!/bin/bash
#
#$ -S /bin/bash
#$ -q long.q
#$ -l ram_free=4G,mem_free=4G,gpu=1

# Example usage of this script:
# EXP_NAME=mid-enblock; export CMD="python train.py --data=$WT_ROOT/pythlm-symlinks --cuda --lr=20.0 --batch_size=20 --bptt=35 --load=$EXP_ROOT/init-lstm-600h-dropout0.5.lm --save=$EXP_ROOT/$EXP_NAME.lm --epochs=25"; qsub -e "$EXP_ROOT/$EXP_NAME.err" -o "$EXP_ROOT/$EXP_NAME.out" -N $EXP_NAME -v CMD smmlm-qsub-wrapper.sh 


# The following just in case an experiment of the same name was run before and log files are the same
echo " ====== New Experiment ======"
echo " ====== New Experiment ======" >&2

unset PYTHONPATH
export PATH=/mnt/matylda5/ibenes/miniconda3/bin:$PATH
unset PYTHONHOME
source activate smmlm

python --version 2>&1
python --version 

ulimit -t unlimited

MACHINE=$(hostname)
echo $MACHINE
echo $MACHINE >&2

echo $CMD
echo $CMD >&2

cd /homes/kazi/ibenes/PhD/pyth-lm
export CUDA_VISIBLE_DEVICES=$(~/util/sge/get-free-gpus.py --max=1)
stdbuf -o0 $CMD
