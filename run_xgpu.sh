# --prefix /d1/scratch/PI/psander/data/VideoLoops
CUDA_VISIBLE_DEVICES=6 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/ustfallclose_mpmesh.txt

#CUDA_VISIBLE_DEVICES=7 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpmesh_sh.txt 1> stdout/out1.txt 2> stdout/out1.txt &