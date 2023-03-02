## Train:

Requires GPU memory >= 24GB, e.g. RTX3090

### stage 1:
```
python train_3d.py --expname fall4_base \
  --config configs/mpi_base.txt --config1 configs/mpis/$DATA_NAME.txt &
```

### stage 2: 
```
python train_3dvid.py --expname fall4_base \
  --config configs/mpv_base.txt --config1 configs/mpvs/$DATA_NAME.txt &
```

### Some implementation details

- The retargeting loss: instead of directly minimizing Q - K