# --prefix /home/mali/VideoLoop3D
CUDA_VISIBLE_DEVICES=1 python train_3d.py --prefix /home/mali/data/VideoLoops \
  --config configs/ustfallclose_mpmesh.txt

#CUDA_VISIBLE_DEVICES=1 python train_3d.py --prefix /home/mali/data/VideoLoops \
#  --config configs/ustfallclose_mpmesh.txt \
#  --expname MESH_ustfallclose720p_dyn_smth02_sparsity0001 --sparsity_loss_weight 0.001
#python train_3dvid.py --prefix /home/mali/VideoLoop3D --config cXonfigs/ustfallclose_mpv_gpnn.txt