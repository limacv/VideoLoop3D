{

########################################################################
# DATA Centeric MPV configs

#for DATASET_NAME in \
#        108fall1narrow_mpvgpnn \
#        108fall2_mpvgpnn 108fall3_mpvgpnn 108fall4_mpvgpnn 108fall5_mpvgpnn \
#        110grasstree_mpvgpnn 110pillar_mpvgpnn
##        1017palm_mpvgpnn 1017yuanrm_mpvgpnn \
##        1020rockrm_mpvgpnn 1020ustfall1_mpvgpnn 1020ustfall2_mpvgpnn \
##        1101grass_mpvgpnn 1101towerd_mpvgpnn \
##        ustfallclose_mpvgpnn usttap_mpvgpnn
#
#do

DATASET_NAME=108fall4_mpvgpnn
TIMECFG=3

echo "processing $DATASET_NAME"

CUDA_VISIBLE_DEVICES=0 python script_render_video.py \
  --config configs/mpvgpnn_wospa.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt --t $TIMECFG &

CUDA_VISIBLE_DEVICES=1 python script_render_video.py \
  --config configs/mpvgpnn_wotv.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt --t $TIMECFG

wait

CUDA_VISIBLE_DEVICES=0 python script_render_video.py \
  --config configs/mpvgpnn_wopyr.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt --t $TIMECFG &

CUDA_VISIBLE_DEVICES=1 python script_render_video.py \
  --config configs/mpvgpnn_wopad.txt --config1 configs/mpvgpnn_final_base/$DATASET_NAME.txt --t $TIMECFG

wait

CUDA_VISIBLE_DEVICES=0 python script_render_video.py \
  --config configs/mpvgpnn_wo1stage.txt --config1 configs/mpvgpnn_final_dense_base/$DATASET_NAME.txt --t $TIMECFG &

CUDA_VISIBLE_DEVICES=1 python script_render_video.py \
  --config configs/mpvloop_dense.txt --config1 configs/mpvloop_final_dense_base/$DATASET_NAME.txt --t $TIMECFG

wait

CUDA_VISIBLE_DEVICES=0 python script_render_video.py \
  --config configs/mpvloop_sparse.txt --config1 configs/mpvloop_final_base/$DATASET_NAME.txt --t $TIMECFG

#wait

wait

#done

  exit
}
