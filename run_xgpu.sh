{
# --prefix /d1/scratch/PI/psander/data/VideoLoops
CUDA_VISIBLE_DEVICES=9 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
  --config configs/ustfallclose_mpmesh.txt

#CUDA_VISIBLE_DEVICES=8 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallfar_mpmesh.txt &

#CUDA_VISIBLE_DEVICES=7 python train_3d.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/usttap_mpmesh.txt &

#CUDA_VISIBLE_DEVICES=4 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops --config configs/ustfallclose_mpv_gpnn.txt

#CUDA_VISIBLE_DEVICES=1 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#  --config configs/ustfallclose_mpv_gpnn.txt \
#  --expname ustfallclose_fp16_v0268GpnnlmA0Gain2P5Pt5S2St2Fnmse_otherGpnnlmA100P3Pt1S2St1Fnmse \
#  --fp16 --lrate 0.01 1> stdout/out1.txt 2> stdout/out1.txt &

#  CUDA_VISIBLE_DEVICES=9 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#    --config configs/ustfallfar_mpv_gpnn.txt &
##
#  CUDA_VISIBLE_DEVICES=8 python train_3dvid.py --prefix /d1/scratch/PI/psander/data/VideoLoops \
#    --config configs/usttap_mpv_gpnn.txt &

  exit
}
