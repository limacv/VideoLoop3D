{


###################################################
# MPI ALL History Configs


#CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/108fall1narrow_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/108fall2_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/108fall3_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/108fall4_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/108fall5_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110grasstree_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/ustfallclose_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=9 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/usttap_mpmesh.txt &
#


#CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017cuhktree_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017dorm1_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017dorm2_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017hair1_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017palm_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017palmbg_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017potflower_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017roadside_mpmesh.txt &
#
#
#
#CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017ustspring_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1017yuan_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1020hydrant_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1020lamp_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1020playground_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1020rock_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1020ustfall1_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/1020ustfall2_mpmesh.txt &



########################################################################
# MPI with configs

# full model
#CFGNAME=mpmesh_shared
#CFG1DIR=mpmesh_final

# dense model
#CFGNAME=mpmesh_dense
#CFG1DIR=mpmesh_final

# wotv model
CFGNAME=mpmesh_wotv
CFG1DIR=mpmesh_final_wotv

CUDA_VISIBLE_DEVICES=9 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpmesh.txt &

CUDA_VISIBLE_DEVICES=8 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpmesh.txt &

CUDA_VISIBLE_DEVICES=7 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpmesh.txt &

CUDA_VISIBLE_DEVICES=6 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpmesh.txt &

CUDA_VISIBLE_DEVICES=5 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpmesh.txt &

CUDA_VISIBLE_DEVICES=4 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpmesh.txt &

CUDA_VISIBLE_DEVICES=3 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillarrm_mpmesh.txt &

CUDA_VISIBLE_DEVICES=2 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpmesh.txt &

wait


CUDA_VISIBLE_DEVICES=9 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017ustspring_mpmesh.txt &

CUDA_VISIBLE_DEVICES=8 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuan_mpmesh.txt &

CUDA_VISIBLE_DEVICES=7 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020hydrant_mpmesh.txt &

CUDA_VISIBLE_DEVICES=6 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rock_mpmesh.txt &

CUDA_VISIBLE_DEVICES=5 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpmesh.txt &

CUDA_VISIBLE_DEVICES=4 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpmesh.txt &

CUDA_VISIBLE_DEVICES=3 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpmesh.txt &

CUDA_VISIBLE_DEVICES=2 python train_3d.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpmesh.txt &

wait 


####################################################################
# Experiment with sparse mpmesh checkpoint

#BASECFG=mpvgpnn_wotv
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt &
#
#wait


#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/usttap_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/1017palm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/1017ustspring_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/1017yuanrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_base/1020ustfall2_mpvgpnn.txt &


#wait


####################################################################
# Experiment with dense mpmesh checkpoint

#BASECFG=mpvgpnn_wo1stage
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/110pillar_mpvgpnn.txt &
#
#wait
#
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/usttap_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/1017palm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/1017ustspring_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/1017yuanrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
#  --config configs/$BASECFG.txt --config1 configs/mpvgpnn_final_dense_base/1020ustfall2_mpvgpnn.txt &



####################################################################
# Experiment with dense mpmesh + LOOP2D

#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/110pillar_mpvgpnn.txt &
#
#wait
#
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/usttap_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/1017palm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/1017ustspring_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/1017yuanrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/1020ustfall2_mpvgpnn.txt &





####################################################################
# Experiment with sparse mpmesh + LOOP2D

#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/110pillar_mpvgpnn.txt &
#
#wait
#
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/usttap_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/1017palm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/1017ustspring_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/1017yuanrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/1020ustfall2_mpvgpnn.txt &



wait

  exit
}
