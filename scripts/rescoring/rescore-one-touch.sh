work_dir=$1
n_jobs=$2

pythlm_model=/mnt/matylda5/ibenes/projects/opensat-2019/pashto/finetune_lr-1.0

kaldi_decode_dir=/mnt/matylda3/karafiat/OpenSAT/2019/Babel_Pashto.v1/exp/chain_MultRDTv1_short/tdnn_lowrank_unconstrained_sp_nleaves2000_new-swbd-cnn/decode_build_dev.seg1_graph_but-web
kaldi_wordlist=/mnt/matylda3/karafiat/OpenSAT/2019/Babel_Pashto.v1/data/lang_test_but-web/words.txt
kaldi_unk='<unk>'
optimal_lmwt=8
kaldi_eg_dir=/mnt/matylda3/karafiat/OpenSAT/2019/Babel_Pashto.v1

# lmws="7.0 7.5 8.0 8.5 9.0"
lmws="9.5 10.0 10.5"
interpolations="0.0 0.05 0.10 0.15 0.20"

# data_dir=$kaldi_eg_dir/data/build_dev.seg1  # contains reference transcription
data_dir=$kaldi_eg_dir/data-hires/build_dev.seg1  # contains reference transcription
graph_dir=$kaldi_eg_dir/exp/chain_MultRDTv1_short/tree_lowrank_unconstrained_sp_nleaves2000/graph_but-web  # who knows what for...
old_lm="fstproject --project_output=true $kaldi_eg_dir/data/lang_test_but-web/G.fst |"

mkdir -p $work_dir

# for ii in $(seq 1 $n_jobs)
# do 
#     lattice-to-nbest \
#             --acoustic-scale=$(echo 1/$optimal_lmwt | bc -l) \
#             --n=100 \
#             "ark:gunzip -c $kaldi_decode_dir/lat.$ii.gz |" ark:- |\
#         tee $work_dir/latt.$ii.nbest |\
#         nbest-to-linear \
#             ark:- \
#             ark,t:$work_dir/$ii.ali \
#             ark,t:$work_dir/$ii.words \
#             ark,t:$work_dir/$ii.hclgscore \
#             ark,t:$work_dir/$ii.acscore || exit 1
#
#     lattice-lmrescore \
#         --lm-scale=-1.0 \
#         "ark:$work_dir/latt.$ii.nbest" \
#         "$old_lm" \
#         ark:- |\
#         nbest-to-linear \
#             ark:- \
#             ark:/dev/null \
#             ark:/dev/null \
#             ark,t:$work_dir/$ii.hclscore \
#             ark:/dev/null || exit 1
#
#     scripts/rescoring/score-combiner.py \
#         $work_dir/$ii.hclgscore 1.0 \
#         $work_dir/$ii.hclscore -1.0 > $work_dir/$ii.gscore
# done

# for ii in $(seq 1 $n_jobs) 
# do 
#     scripts/rescoring/rescore-kaldi-latt.py \
#         --latt-vocab=$kaldi_wordlist \
#         --latt-unk=$kaldi_unk \
#         --model-from=$pythlm_model \
#         $work_dir/$ii.words \
#         $work_dir/$ii.rnnlm-scores 
# done

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
echo PWD: $PWD

export KALDI_ROOT=/mnt/matylda3/karafiat/BABEL/GIT/Kaldi.cur
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

export LC_ALL=C
export CUDA_CACHE_DISABLE=1


# for pickdir in $work_dir/pick-orig-lattices-like-martas
# for pickdir in $work_dir/pick-orig-lattices 
for pickdir in  $work_dir/pick-1.0-* 
do 
    echo $n_jobs > $pickdir/num_jobs
    local/score.sh --min_lmwt 8 --max_lmwt 8 $data_dir $graph_dir $pickdir
done

# cat $work_dir/pick-1.0-*/scoring_kaldi/best_wer | grep -v 'nan' | sort -k2 -n > $work_dir/all_wers
grep 'Error' $work_dir/pick-1.0-8-*/scoring_8/ctm_filt_stm.dtl > $work_dir/all_wers

