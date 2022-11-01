{

#CFGNAME=mpvgpnn_wo1stage
#CFG1DIR=mpvgpnn_final_dense_base
#
#CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=3 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillar_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpvgpnn.txt &
#
#
#wait
#
#CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017ustspring_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuanrm_mpvgpnn.txt &
#
##CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
##  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020hydrant_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rockrm_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=1 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpvgpnn.txt &
#
#CUDA_VISIBLE_DEVICES=2 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpvgpnn.txt &




#####################################################################
# LOOPING
CFGNAME=mpvloop_sparse
CFG1DIR=mpvloop_final_eval

CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall1narrow_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall2_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall3_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall4_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/108fall5_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110grasstree_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=3 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/110pillar_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=2 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017palm_mpvgpnn.txt &


wait

CUDA_VISIBLE_DEVICES=9 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017ustspring_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=8 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1017yuanrm_mpvgpnn.txt &

#CUDA_VISIBLE_DEVICES=7 python script_evaluate_ours.py \
#  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020hydrant_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=6 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020rockrm_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=5 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall1_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=4 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/1020ustfall2_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=1 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/ustfallclose_mpvgpnn.txt &

CUDA_VISIBLE_DEVICES=2 python script_evaluate_ours.py \
  --config configs/$CFGNAME.txt --config1 configs/$CFG1DIR/usttap_mpvgpnn.txt &

wait

  exit
}
