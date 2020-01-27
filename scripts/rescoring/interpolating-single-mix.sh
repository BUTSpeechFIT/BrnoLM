work_dir=$1
ac_w=$2
hcl_w=$3
lm_w=$4
interpolation_nnlm=$5
n_jobs=$6

interpolation_ngram=$(python -c "print(1.0 - $interpolation_nnlm)")

pick_dir="pick-$ac_w-$hcl_w-$lm_w-$interpolation_nnlm"

mkdir $work_dir/$pick_dir || { echo "directory $work_dir/$pick_dir already exists" >&2 ; exit 1 ; }

for ii in $(seq 1 $n_jobs) ; do 
    scripts/rescoring/score-combiner.py \
        $work_dir/$ii.gscore $interpolation_ngram \
        $work_dir/$ii.rnnlm-scores $interpolation_nnlm > $work_dir/$pick_dir/$ii.combined-lm-scores

    scripts/rescoring/rescoring-combine-scores.py \
        --ac-scale=$ac_w \
        --gr-scale=$hcl_w \
        --lm-scale=$lm_w \
        $work_dir/$ii.acscore \
        $work_dir/$ii.hclscore \
        $work_dir/$pick_dir/$ii.combined-lm-scores \
        $work_dir/$pick_dir/$ii.pick 

    lattice-copy ark:$work_dir/latt.$ii.nbest ark,t:- |\
        scripts/rescoring/pick-best.py $work_dir/$pick_dir/$ii.pick |\
        lattice-copy ark,t:- ark:- |\
        gzip -c > $work_dir/$pick_dir/lat.$ii.gz 
done
