{
########################################################################
# Customized settings

CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_spa0001 --init_from meshlog1final/110pillar_spa0001/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_spa0 --init_from meshlog1final/110pillar_spa0/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_spa0012 --init_from meshlog1final/110pillar_spa0012/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_spa002 --init_from meshlog1final/110pillar_spa002/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_spa0002 --init_from meshlog1final/110pillar_spa0002/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_spa0008 --init_from meshlog1final/110pillar_spa0008/epoch_0119.tar &





########################################################################
# DATA Centeric MPV configs

#DATASET_NAME=1101towerd_mpvgpnn
#
#CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
#  --config configs/mpvgpnn_wospa.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
#  --config configs/mpvgpnn_wotv.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
#  --config configs/mpvgpnn_wopyr.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
#  --config configs/mpvgpnn_wopad.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
#  --config configs/mpvgpnn_wo1stage.txt --config1 configs/mpvgpnn_final_dense_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=3 python script_evaluate_ours.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/$DATASET_NAME.txt &

#wait

#########################################################################
# Config centric

# full # -- train ok
#CFGNAME=mpvgpnn_wospa1
#CFG1DIR=mpvgpnn_final_base

# ablation, wo tv -- train ok
#CFGNAME=mpvgpnn_wotv
#CFG1DIR=mpvgpnn_final_base_wotv

# ablation, wo pyr -- train ok
#CFGNAME=mpvgpnn_wopyr
#CFG1DIR=mpvgpnn_final_base

# ablation, wo pad -- train ok
#CFGNAME=mpvgpnn_wopad
#CFG1DIR=mpvgpnn_final_base

# ablation, wo 1 stage -- train ok
#CFGNAME=mpvgpnn_wo1stage
#CFG1DIR=mpvgpnn_final_dense_base

# experiment, loop with dense MPI -- train ok
#CFGNAME=mpvloop_dense
#CFG1DIR=mpvloop_final_dense_base

# experiment, loop with our sparse representation  -- train ok
#CFGNAME=mpvloop_sparse
#CFG1DIR=mpvloop_final_base


#CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillar_mpvgpnn.txt &
#
#
#wait
#
#CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuanrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpvgpnn.txt &



wait

  exit
}
