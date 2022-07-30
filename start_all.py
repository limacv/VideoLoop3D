import os
from glob import glob

files = [
    "droplet_mpv_fromNoise"
    # "simplevid_pinatas4",
    # "simplevid_squareflags3"
    # "simplevid_brink2down_nopyr"
]

files = [file + ".txt" for file in files]

startjson = """
{
  "Token": "loCr7k16k3DEY7283MuFqA",
  "business_flag": "TEG_AILab_CVC_DigitalContent",
  "model_local_file_path": "/apdcephfs/private_leema/VideoLoop3D",
  "host_num": 1,
  "host_gpu_num": <gpu_num>,
  "GPUName": "A100,V100",
  "image_full_name": "mirrors.tencent.com/leema/svox2:0",
  "init_cmd": "jizhi_client mount -l cq ~/private_leema",
  "start_cmd": "conda activate /apdcephfs/private_leema/Environments/pt3d; python train_3dvid.py --config configs/<file_base>",
  "task_flag": "<task_name>"
}
"""
for file in files:
    with open(os.path.join("configs", file), 'r') as f:
        lines = f.read().splitlines()
    line = [l_ for l_ in lines if "expname" in l_][0]
    line = line[line.find("=") + 1:]
    line = line.lstrip(' ').strip('\n')

    gpuline = [l_ for l_ in lines if "gpu_num" in l_][0]
    gpuline = gpuline[gpuline.find("=") + 1:]
    gpuline = gpuline.lstrip(' ').strip('\n')
    gpuline = int(gpuline)
    print(f"Launching expname = {line} with gpunum = {gpuline}")
    with open("_tmp_start.json", 'w') as f:
        json = startjson.replace("<gpu_num>", f"{gpuline}")\
            .replace("<file_base>", file)\
            .replace("<task_name>", line)
        f.write(json)
    os.system("jizhi_client start -scfg _tmp_start.json")
