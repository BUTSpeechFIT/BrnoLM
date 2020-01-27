work_dir=$1
n_jobs=$2

pythlm_model=/mnt/matylda5/ibenes/projects/wsj-rescoring/new-lm.lm

kaldi_decode_dir=/mnt/matylda5/ibenes/kaldi/egs/wsj/s5/exp/tri4b/decode_bd_tgpr_dev93
kaldi_wordlist=/mnt/matylda5/ibenes/kaldi/egs/wsj/s5/data/lang_test_bd_tgpr/words.txt
kaldi_unk='<UNK>'
optimal_lmwt=17.0
kaldi_eg_dir=/mnt/matylda5/ibenes/kaldi/egs/wsj/s5

lmws="15.5 16.5 17.5 18.5 19.5"
interpolations="0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40"

data_dir=data/test_dev93 #  contains reference transcription
graph_dir=exp/tri4b/graph_bd_tgpr #  who knows what for...
old_lm="fstproject --project_output=true $kaldi_eg_dir/data/lang_test_bd_tgpr/G.fst |"

mkdir -p $work_dir

for ii in $(seq 1 $n_jobs)
do 
    lattice-to-nbest \
            --acoustic-scale=$(echo 1/$optimal_lmwt | bc -l) \
            --n=100 \
            "ark:gunzip -c $kaldi_decode_dir/lat.$ii.gz |" ark:- |\
        tee $work_dir/latt.$ii.nbest |\
        nbest-to-linear \
            ark:- \
            ark,t:$work_dir/$ii.ali \
            ark,t:$work_dir/$ii.words \
            ark,t:$work_dir/$ii.hclgscore \
            ark,t:$work_dir/$ii.acscore || break

    lattice-lmrescore \
        --lm-scale=-1.0 \
        "ark:$work_dir/latt.$ii.nbest" \
        "$old_lm" \
        ark:- |\
        nbest-to-linear \
            ark:- \
            ark:/dev/null \
            ark:/dev/null \
            ark,t:$work_dir/$ii.hclscore \
            ark:/dev/null || break

    scripts/rescoring/score-combiner.py \
        $work_dir/$ii.hclgscore 1.0 \
        $work_dir/$ii.hclscore -1.0 > $work_dir/$ii.gscore
done

for ii in $(seq 1 $n_jobs) 
do 
    scripts/rescoring/rescore-kaldi-latt.py \
        --latt-vocab=$kaldi_wordlist \
        --latt-unk=$kaldi_unk \
        --model-from=$pythlm_model \
        $work_dir/$ii.words \
        $work_dir/$ii.rnnlm-scores 
done

for lm_w in $lmws
do
    for interpolation_lambdas in $interpolations
    do
        scripts/rescoring/interpolating-single-mix.sh $work_dir \
            1.0 $optimal_lmwt $lm_w $interpolation_lambdas \
            $n_jobs 
    done
done

cd $kaldi_eg_dir
. path.sh

for pickdir in $work_dir/pick-1.0-* 
do 
    local/score.sh --min_lmwt 1 --max_lmwt 1 $data_dir $graph_dir $pickdir
done

cat $work_dir/pick-1.0-*/scoring_kaldi/best_wer | grep -v 'nan' | sort -k2 -n > $work_dir/all_wers

