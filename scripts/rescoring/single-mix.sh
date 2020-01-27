work_dir=$1
ac_w=$2
hclg_w=$3
nnlm_w=$4
n_jobs=$5

pick_dir="pick-$ac_w-$hclg_w-$nnlm_w"

mkdir $work_dir/$pick_dir || { echo "directory $work_dir/$pick_dir already exists" >&2 ; exit 1 ; }

for ii in $(seq 1 $n_jobs) ; do 
    scripts/rescoring/rescoring-combine-scores.py \
        --ac-scale=$ac_w \
        --gr-scale=$hclg_w \
        --lm-scale=$nnlm_w \
        $work_dir/$ii.acscore \
        $work_dir/$ii.hclgscore \
        $work_dir/$ii.rnnlm-scores \
        $work_dir/$pick_dir/$ii.pick 

    lattice-copy ark:$work_dir/latt.$ii.nbest ark,t:- |\
        scripts/rescoring/pick-best.py $work_dir/$pick_dir/$ii.pick |\
        lattice-copy ark,t:- ark:- |\
        gzip -c > $work_dir/$pick_dir/lat.$ii.gz 
done
