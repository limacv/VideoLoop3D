{

########################################################################
# DATA Centeric MPI configs

DATASET_NAME=1101towerd_mpmesh

CUDA_VISIBLE_DEVICES=9 python train_3d.py \
  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/mpmesh_dense.txt --config1 configs/mpmesh_final/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_wotv.txt --config1 configs/mpmesh_final_wotv/$DATASET_NAME.txt &

#wait

########################################################################
# MPI with configs

# full model
#CFGNAME=mpmesh_shared
#CFG1DIR=mpmesh_final

# dense model
#CFGNAME=mpmesh_dense
#CFG1DIR=mpmesh_final

# wotv model
#CFGNAME=mpmesh_wotv
#CFG1DIR=mpmesh_final_wotv

#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillarrm_mpmesh.txt &
#
#wait
#
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuan_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rock_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpmesh.txt &
#
#wait


#CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101grass_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101towerd_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustpalm_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustyard_mpmesh.txt &

#wait
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################


########################################################################
# DATA Centeric MPV configs

#DATASET_NAME=108fall1narrow_mpvgpnn

#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/mpvgpnn_wospa.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvgpnn_wotv.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/mpvgpnn_wopyr.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/mpvgpnn_wopad.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/mpvgpnn_wo1stage.txt --config1 configs/mpvgpnn_final_dense_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/$DATASET_NAME.txt &

#wait

####################################################################
# MPV with sparse representation

# used for visualization -- todo
#CFGNAME=mpvgpnn_notest
#CFG1DIR=mpvgpnn_final_base_notest

# full -- train ok
#CFGNAME=mpvgpnn_wospa
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


wait

  exit
}
