{

#############################################
# Run all the 1st stage
#############################################

CUDA_VISIBLE_DEVICES=0 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/108fall1.txt &
CUDA_VISIBLE_DEVICES=1 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/108fall2.txt &
CUDA_VISIBLE_DEVICES=2 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/108fall3.txt &
CUDA_VISIBLE_DEVICES=3 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/108fall4.txt &
CUDA_VISIBLE_DEVICES=4 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/108fall5.txt &
CUDA_VISIBLE_DEVICES=5 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/110grasstree.txt &
CUDA_VISIBLE_DEVICES=6 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/110pillarrm.txt &
CUDA_VISIBLE_DEVICES=7 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1017palm.txt &

wait

CUDA_VISIBLE_DEVICES=8 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1017yuan.txt &
CUDA_VISIBLE_DEVICES=7 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1020rock.txt &
CUDA_VISIBLE_DEVICES=6 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1020ustfall1.txt &
CUDA_VISIBLE_DEVICES=5 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1020ustfall2.txt &
CUDA_VISIBLE_DEVICES=4 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1101grass.txt &
CUDA_VISIBLE_DEVICES=3 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/1101towerd.txt &
CUDA_VISIBLE_DEVICES=1 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/ustfallclose.txt &
CUDA_VISIBLE_DEVICES=0 python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/usttap.txt &

wait

#############################################
# Run all the 2nd stage
#############################################

CUDA_VISIBLE_DEVICES=0 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall1.txt &
CUDA_VISIBLE_DEVICES=1 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall2.txt &
CUDA_VISIBLE_DEVICES=2 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall3.txt &
CUDA_VISIBLE_DEVICES=3 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall4.txt &
CUDA_VISIBLE_DEVICES=4 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall5.txt &
CUDA_VISIBLE_DEVICES=5 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/110grasstree.txt &
CUDA_VISIBLE_DEVICES=6 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/110pillar.txt &
CUDA_VISIBLE_DEVICES=7 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1017palm.txt &


wait

CUDA_VISIBLE_DEVICES=8 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1017yuan.txt &
CUDA_VISIBLE_DEVICES=7 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1020rock.txt &
CUDA_VISIBLE_DEVICES=6 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1020ustfall1.txt &
CUDA_VISIBLE_DEVICES=5 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1020ustfall2.txt &
CUDA_VISIBLE_DEVICES=4 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1101grass.txt &
CUDA_VISIBLE_DEVICES=3 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/1101towerd.txt &
CUDA_VISIBLE_DEVICES=1 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/ustfallclose.txt &
CUDA_VISIBLE_DEVICES=0 python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/usttap.txt &


#############################################
#export all meshes
#############################################

python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall1.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall2.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall3.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall4.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/108fall5.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/110grasstree.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/110pillar.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1017palm.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1017yuan.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1020rock.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1020ustfall1.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1020ustfall2.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1101grass.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/1101towerd.txt
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/ustfallclose.txt &
python script_export_mesh.py --config configs/mpv_base.txt --config1 configs/mpvs/usttap.txt



wait

  exit
}
