{
#CUDA_VISIBLE_DEVICES=3 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpmesh.txt
#
#CUDA_VISIBLE_DEVICES=8 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallfar_mpmesh.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/usttap_mpmesh.txt &

#CUDA_VISIBLE_DEVICES=1 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallfar_mpmesh.txt \
#  --sparsify_rmfirstlayer --expname ustfallfar720p32layer_dyn_smth02_sparsity0004_cullloop_rm1st &
#
#CUDA_VISIBLE_DEVICES=0 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/usttap_mpmesh.txt \
#  --sparsify_rmfirstlayer --expname usttap720p32layer_dyn_smth02_sparsity0004_cullloop_rm1st &

#CUDA_VISIBLE_DEVICES=9 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn1.txt &
#
#CUDA_VISIBLE_DEVICES=7 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn2.txt &
#
#CUDA_VISIBLE_DEVICES=6 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn3.txt &

#CUDA_VISIBLE_DEVICES=5 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn4.txt &
#
#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn5.txt &




CUDA_VISIBLE_DEVICES=9 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108elevator_mpmesh.txt &

CUDA_VISIBLE_DEVICES=8 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108fall1_mpmesh.txt &

CUDA_VISIBLE_DEVICES=7 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108fall2_mpmesh.txt &

CUDA_VISIBLE_DEVICES=6 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108fall3_mpmesh.txt &

CUDA_VISIBLE_DEVICES=5 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108fall4_mpmesh.txt &

CUDA_VISIBLE_DEVICES=4 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108fall5_mpmesh.txt &

CUDA_VISIBLE_DEVICES=3 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108fall31_mpmesh.txt &

CUDA_VISIBLE_DEVICES=2 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108hole_mpmesh.txt &

CUDA_VISIBLE_DEVICES=1 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108sit_mpmesh.txt &

CUDA_VISIBLE_DEVICES=0 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108spring1_mpmesh.txt &

CUDA_VISIBLE_DEVICES=1 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108spring3_mpmesh.txt &

CUDA_VISIBLE_DEVICES=2 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108waterstair_mpmesh.txt &

CUDA_VISIBLE_DEVICES=3 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/108waving_mpmesh.txt &


  exit
}
