# 3D Video Loops from Asynchronous Input
This repository is the official code for the CVPR23 paper: **3D Video Loops from Asynchronous Input**. Please visit our [project page](https://limacv.github.io/VideoLoop3D_web/) for more information, such as supplementary, demo and dataset.

## Introduction
In this project, we construct a 3D video loop from multi-view videos that can be asynchronous. The 3D video loop is represented as MTV, a new representation, which is essentially multiple tiles with dynamic textures. This code implements the following functionality: 

1. The 2-stage optimization, which is the core of the paper.
2. An off-line renderer that render using pytorch slowly.
3. Evaluation code that compute metrics for comparison.
4. Scripts for data preprocessing and mesh export.

There is another WebGL based renderer implemented [here](https://github.com/limacv/VideoLoopUI), which renders the exported mesh in real time even on an old iPhone X.

## Train on dataset

### prerequisite

- The optimization is quite memory consuming. It requires a GPU with memory >= 24GB, e.g. RTX3090. Make sure you have enough GPU memory!
- Install dependencies in the ```requirements.txt```
```
conda create -n vloop3d python==3.8
conda activate vloop3d
pip install -r requirements.txt
```

### dataset
Download dataset from the link [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lmaag_connect_ust_hk/EiZnIyUYmJdLpQ5hiLtkz8IBfAyeoUiXHt5H0-pFgzV9cg?e=OBNIas). Place them somewhere. For example, you've placed ```fall2720p``` in ```<data_dir>/fall2720p```.

### config path
In the ```configs/mpi_base.txt``` and ```configs/mpv_base.txt```, change the ```prefix``` dir to ```<data_dir>```. 

Then later all the files will be stored in the ```<prefix>/<expdir>/<expname>```. In the example it will be ```<data_dir>/mpis/108fall2``` and ```<data_dir>/mpvs/108fall2```.

### stage 1:
In this stage, we generate static Multi-plane Image (MPI) and 3D loopable mask (typically 10-15mins).
Run following:
```
python train_3d.py --config configs/mpi_base.txt --config1 configs/mpis/$DATA_NAME.txt
```

### stage 2:
After stage 1 finishes, run following. Note this will load **epoch_0119.tar** file generated in stage 1. In stage 2, we generate final 3D looping video using looping loss (typically 3-6h).
```
python train_3dvid.py --config configs/mpv_base.txt --config1 configs/mpvs/$DATA_NAME.txt
```

After stage 2 finishes, you can get a 3D video loop saved as *.tar file.

## Evaluation

TODO

## Export mesh

TODO

## Using your own data

TODO

## Other Notes

- The retargeting loss: instead of directly minimizing Q - K
- In each iteration, we randomly perturb the camera intrinsic for half pixel (i.e. cx += rand() - 0.5, same for cy). We find this can reduce the tiling artifact. See the demo [here](https://limacv.github.io/VideoLoopUI/?dataurl=assets/ustfall1_tiling) for adding this perturb and [here](https://limacv.github.io/VideoLoopUI/?dataurl=assets/ustfall1) for without perturb. There is still some artifact when render in high resolution (the training is conducted in 640x360).