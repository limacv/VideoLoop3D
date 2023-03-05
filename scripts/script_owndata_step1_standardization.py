import cv2
import argparse
import imageio
import os
import numpy as np

parser = argparse.ArgumentParser(description='davince to my')
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_prefix', default="../data", help="Where to put the results")
parser.add_argument('--factor', default="1,2", help="factors")
args = parser.parse_args()

# args = argparse.Namespace(
#     input_path = "../data/1017ustspring.mov",
#     output_prefix = "../data/1017ustspring720p/",
#     factor = [1, 2],
# )

if isinstance(args.factor, str):
    args.factor = list(map(int, args.factor.split(',')))
print(f"Saving to {args.output_prefix}")


def saving2prefix(frames, prefix, factors):
    avg_img = np.array(frames)
    avg_img = np.mean(avg_img, 0)
    avg_img = avg_img.astype(np.uint8)
    avg_outp = prefix + f"/images/{clip_id:04d}.png"
    os.makedirs(os.path.dirname(avg_outp), exist_ok=True)
    imageio.imwrite(avg_outp, avg_img)

    for factor in factors:
        vid_outp = prefix + f"/videos_{factor}/{clip_id:04d}.mp4"
        os.makedirs(os.path.dirname(vid_outp), exist_ok=True)
        images = [cv2.resize(im, None, None, 1 / factor, 1 / factor) for im in frames]
        imageio.mimwrite(vid_outp, images, fps=25, macro_block_size=1, quality=8)


cap = cv2.VideoCapture(args.input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
factors = args.factor
if isinstance(factors, (int, float)):
    factors = [factors]

clip_id = 0
images = []
imgsums = []
while True:
    ret, img = cap.read()
    if not ret:
        if len(images) > 0:  # saving
            saving2prefix(images, args.output_prefix, factors)
            print(f'saving videos of frame {len(images)}')
            clip_id += 1
        break

    sum_ = img.mean()
    imgsums.append(sum_)
    if sum_ < 10:  # new sequence
        if len(images) > 0:  # saving
            saving2prefix(images, args.output_prefix, factors)
            print(f'saving videos of frame {len(images)}')
            clip_id += 1
        # reinitialize
        images = []
    else:
        images.append(img[..., ::-1])

import matplotlib.pyplot as plt

plt.plot(imgsums)
plt.show()
