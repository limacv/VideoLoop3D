{


########################################################################
# Customized configs


#CUDA_VISIBLE_DEVICES=9 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
#  --expname 110pillar_spa0001 --sparsity_loss_weight 0.001 &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
#  --expname 110pillar_spa0 --sparsity_loss_weight 0 &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
#  --expname 110pillar_spa0012 --sparsity_loss_weight 0.01 &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
#  --expname 110pillar_spa002 --sparsity_loss_weight 0.02 &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
#  --expname 110pillar_spa0002 --sparsity_loss_weight 0.002 &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
#  --expname 110pillar_spa0008 --sparsity_loss_weight 0.008 &
#
#wait
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
#  --expname sparse_110pillar_spa0001 --init_from meshlog1final/110pillar_spa0001/epoch_0119.tar &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
#  --expname sparse_110pillar_spa0 --init_from meshlog1final/110pillar_spa0/epoch_0119.tar &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
#  --expname sparse_110pillar_spa0012 --init_from meshlog1final/110pillar_spa0012/epoch_0119.tar &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
#  --expname sparse_110pillar_spa002 --init_from meshlog1final/110pillar_spa002/epoch_0119.tar &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
#  --expname sparse_110pillar_spa0002 --init_from meshlog1final/110pillar_spa0002/epoch_0119.tar &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
#  --expname sparse_110pillar_spa0008 --init_from meshlog1final/110pillar_spa0008/epoch_0119.tar &



CUDA_VISIBLE_DEVICES=9 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv02 --rgb_smooth_loss_weight 0.07 --a_smooth_loss_weight 0.2 &

CUDA_VISIBLE_DEVICES=8 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv01 --rgb_smooth_loss_weight 0.033 --a_smooth_loss_weight 0.1 &

CUDA_VISIBLE_DEVICES=7 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv005 --rgb_smooth_loss_weight 0.02 --a_smooth_loss_weight 0.05 &

CUDA_VISIBLE_DEVICES=6 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv001 --rgb_smooth_loss_weight 0.0033 --a_smooth_loss_weight 0.01 &

CUDA_VISIBLE_DEVICES=5 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv075 --rgb_smooth_loss_weight 0.25 --a_smooth_loss_weight 0.75 &

CUDA_VISIBLE_DEVICES=4 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv1 --rgb_smooth_loss_weight 0.33 --a_smooth_loss_weight 1 &

CUDA_VISIBLE_DEVICES=3 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv15 --rgb_smooth_loss_weight 0.5 --a_smooth_loss_weight 1.5 &

CUDA_VISIBLE_DEVICES=2 python train_3d.py --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/110pillarrm_mpmesh.txt \
  --expname 110pillar_tv24 --rgb_smooth_loss_weight 0.8 --a_smooth_loss_weight 2.4 &

wait

CUDA_VISIBLE_DEVICES=9 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv02 --init_from meshlog1final/110pillar_tv02/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=8 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv01 --init_from meshlog1final/110pillar_tv01/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=7 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv005 --init_from meshlog1final/110pillar_tv005/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=6 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv001 --init_from meshlog1final/110pillar_tv001/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=5 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv075 --init_from meshlog1final/110pillar_tv075/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=4 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv1 --init_from meshlog1final/110pillar_tv1/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=3 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv15 --init_from meshlog1final/110pillar_tv15/epoch_0119.tar &

CUDA_VISIBLE_DEVICES=2 python train_3dvid.py --config configs/mpvgpnn_abspa.txt --config1 configs/mpvgpnn_final_base/110pillar_mpvgpnn.txt \
  --expname sparse_110pillar_tv24 --init_from meshlog1final/110pillar_tv24/epoch_0119.tar &



########################################################################
# DATA Centeric MPI configs

#DATASET_NAME=108fall4_mpmesh

#CUDA_VISIBLE_DEVICES=9 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_final/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/mpmesh_dense.txt --config1 configs/mpmesh_final/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_wotv.txt --config1 configs/mpmesh_final_wotv/$DATASET_NAME.txt &

wait

########################################################################
# MPI with configs

# full model
#CFGNAME=mpmesh_shared
#CFG1DIR=mpmesh_final
#FLAG=""

# dense model
#CFGNAME=mpmesh_dense
#CFG1DIR=mpmesh_final

# wotv model
#CFGNAME=mpmesh_wotv
#CFG1DIR=mpmesh_final_wotv

#CFGNAME=mpmesh_shared
#CFG1DIR=mpmesh_final
#FLAG="--density_loss_weight 0 --expdir meshlog_wodensity"
#
#CUDA_VISIBLE_DEVICES=9 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillarrm_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpmesh.txt $FLAG &
#
#wait
#
#
#CUDA_VISIBLE_DEVICES=9 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuan_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rock_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101grass_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101towerd_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustpalm_mpmesh.txt $FLAG &
#
#CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustyard_mpmesh.txt $FLAG &
#
#wait


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################


########################################################################
# DATA Centeric MPV configs

#DATASET_NAME=108fall4_mpvgpnn
#
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/mpvgpnn_wospa.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvgpnn_wotv.txt --config1 configs/mpvgpnn_final_base_wotv/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/mpvgpnn_wopyr.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/mpvgpnn_wopad.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/mpvgpnn_wo1stage.txt --config1 configs/mpvgpnn_final_dense_base/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/$DATASET_NAME.txt &

#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/$DATASET_NAME.txt &
#
#wait

####################################################################
# MPV with sparse representation

# used for visualization
#CFGNAME=mpvgpnn_woden
#CFG1DIR=mpvgpnn_final_base_woden

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


#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillar_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101grass_mpvgpnn.txt &


wait
#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101towerd_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuanrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpvgpnn.txt &


#CUDA_VISIBLE_DEVICES=1 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustpalm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=0 python train_3dvid.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustyard_mpvgpnn.txt &

wait

  exit
}
