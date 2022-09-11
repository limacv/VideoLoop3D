# dryrun
#echo "Start Dry Run"
#python train_3dvid.py --config configs/droplet_mpv_std01.txt --pyr_num_epoch 1
#echo "===================Finish Dry Run================="

#python train_3d.py --config configs/droplet_mpmesh.txt
python train_3dvid.py --config configs/droplet_mpv.txt --i_print 99999999
