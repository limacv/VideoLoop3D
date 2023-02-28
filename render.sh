#{

########################################################################
# DATA Centeric MPV configs

#for DATASET_NAME in \
#        108fall1narrow_mpvgpnn \
#        108fall2_mpvgpnn 108fall3_mpvgpnn 108fall4_mpvgpnn 108fall5_mpvgpnn \
#        110grasstree_mpvgpnn 110pillar_mpvgpnn \
#        1017palm_mpvgpnn 1017yuanrm_mpvgpnn \
#        1020rockrm_mpvgpnn 1020ustfall1_mpvgpnn 1020ustfall2_mpvgpnn \
#        1101grass_mpvgpnn 1101towerd_mpvgpnn \
#        ustfallclose_mpvgpnn usttap_mpvgpnn
#
#do

DATASET_NAME=1017dorm1_mpvgpnn
#TIMECFG="--t 0"

#echo "processing $DATASET_NAME"

CUDA_VISIBLE_DEVICES=0 python script_render_video.py \
  --config configs/mpvgpnn_wospa.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt --t 0 --v 1 &

#CUDA_VISIBLE_DEVICES=1 python script_render_video.py \
#  --config configs/mpvgpnn_wotv.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt
#
#CUDA_VISIBLE_DEVICES=2 python script_render_video.py \
#  --config configs/mpvgpnn_wopyr.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=3 python script_render_video.py \
#  --config configs/mpvgpnn_wopad.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt
#
#CUDA_VISIBLE_DEVICES=4 python script_render_video.py \
#  --config configs/mpvgpnn_wo1stage.txt --config1 configs/mpvgpnn_final_dense_base/$DATASET_NAME.txt &
#
#CUDA_VISIBLE_DEVICES=5 python script_render_video.py \
#  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/$DATASET_NAME.txt
#
#CUDA_VISIBLE_DEVICES=6 python script_render_video.py \
#  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/$DATASET_NAME.txt

#wait

#done

#  exit
#}


{

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


#CUDA_VISIBLE_DEVICES=9 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpvgpnn.txt --v r0 --f 150 --render_scaling 0.7 &
#
#CUDA_VISIBLE_DEVICES=8 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpvgpnn.txt --v r0 --f 150 &
#
#CUDA_VISIBLE_DEVICES=7 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpvgpnn.txt --v r0 --f 150 --render_scaling 1.2 &
#
#CUDA_VISIBLE_DEVICES=6 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpvgpnn.txt --v r0 --f 150 --render_scaling 1.2  &
#
#CUDA_VISIBLE_DEVICES=5 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpvgpnn.txt --v r0 --f 150 --render_scaling 1.2 &
#
#CUDA_VISIBLE_DEVICES=4 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpvgpnn.txt --v r0 --f 150 --render_scaling 0.6 &
#
#CUDA_VISIBLE_DEVICES=3 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillar_mpvgpnn.txt --v r0 --f 150 --render_scaling 1.4 &
#
#CUDA_VISIBLE_DEVICES=2 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101grass_mpvgpnn.txt --v r0 --f 150 --render_scaling 1.4 &
#
#
##wait
#CUDA_VISIBLE_DEVICES=9 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101towerd_mpvgpnn.txt --v r0 --f 180 &
#
#CUDA_VISIBLE_DEVICES=8 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpvgpnn.txt --v r0 --f 180 &
#
#CUDA_VISIBLE_DEVICES=7 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpvgpnn.txt --v r0 --f 180 --render_scaling 0.8 &
#
#CUDA_VISIBLE_DEVICES=6 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpvgpnn.txt --v r0 --f 180 --render_scaling 1.4 &
#
#CUDA_VISIBLE_DEVICES=5 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuanrm_mpvgpnn.txt --v r0 --f 150 &
#
#CUDA_VISIBLE_DEVICES=4 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rockrm_mpvgpnn.txt --v r0 --f 150 &
#
#CUDA_VISIBLE_DEVICES=3 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpvgpnn.txt --v r0 --f 180 &
#
#CUDA_VISIBLE_DEVICES=2 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpvgpnn.txt --v r0 --f 150 &


wait
  exit
}

#CUDA_VISIBLE_DEVICES=1 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustpalm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=0 python script_render_video.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1101ustyard_mpvgpnn.txt &
