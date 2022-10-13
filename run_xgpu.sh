{

CUDA_VISIBLE_DEVICES=9 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall1_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall1narrow_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall2_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall3_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=5 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall4_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall5_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/108fall31_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=2 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/109duoelevator_mpvgpnn.txt &



CUDA_VISIBLE_DEVICES=1 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/109fence_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=3 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/109tree_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=8 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/110grasstree_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=7 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/110pillar_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=6 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/ustfallclose_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=4 python train_3dvid.py \
  --config configs/mpvgpnn_shared.txt --config1 configs/mpvgpnn_fromdensity/usttap_mpvgpnn.txt &


#####################################################
# MPI
#
#CUDA_VISIBLE_DEVICES=9 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall1_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall2_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall3_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall4_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall5_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall31_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/109duoelevator_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/109fence_mpmesh.txt &


#CUDA_VISIBLE_DEVICES=8 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/108fall1narrow_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/109tree_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/110grasstree_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=5 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/110pillar_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/ustfallclose_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=3 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/ustfallfar_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=2 python train_3d.py \
#  --config configs/mpmesh_shared.txt --config1 configs/mpmesh_density/usttap_mpmesh.txt &

  exit
}
