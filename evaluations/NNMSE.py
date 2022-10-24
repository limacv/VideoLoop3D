from utils_vid import extract_3Dpatches, get_NN_indices_low_memory
import warnings
import torch
import numpy as np


def compute_nnerr(src, tar,
                  patch_size=7, stride=2, patcht_size=7, stridet=2,
                  macro_block=65):
        """
        x, tar: shape of B x 3 x f x h x w
        """
        # standardlize the input
        t, h, w = src.shape[-3:]

        def fit_patch(s_, name, p_, st_):
            if (s_ - p_) % st_ != 0:
                new_s_ = (s_ - p_) // st_ * st_ + p_
                warnings.warn(f'{name} doesnot satisfy ({name} - patch_size) % stride == 0. '
                              f'changing {name} from {s_} to {new_s_}')
                return new_s_
            return s_

        macro_block = fit_patch(macro_block, "macro_block", patch_size, stride)
        h = fit_patch(h, "patch_height", patch_size, stride)
        w = fit_patch(w, "patch_width", patch_size, stride)
        t = fit_patch(t, "frame_num", patcht_size, stridet)
        src = src[..., :t, :h, :w]
        tar = tar[..., :h, :w]

        with torch.no_grad():
            macro_stride = macro_block - patch_size + stride
            h_starts = np.arange(0, h - macro_block + macro_stride, macro_stride)
            w_starts = np.arange(0, w - macro_block + macro_stride, macro_stride)
            errs = []
            for h_start in h_starts:
                # if h - h_start < patch_size:  # this checking is nolonger needed due to the fit_patch
                #     h_start -= patch_size
                for w_start in w_starts:
                    # if w - w_start < patch_size:
                    #     w_start -= patch_size
                    src_crop = src[..., h_start: h_start + macro_block, w_start: w_start + macro_block]
                    tar_crop = tar[..., h_start: h_start + macro_block, w_start: w_start + macro_block]
                    # partation input into different patches and process individually
                    projsrc = extract_3Dpatches(src_crop, patch_size, patcht_size, stride, stridet)  # b, c, d, h, w
                    b, c, d, h, w = projsrc.shape
                    B = b * h * w
                    D = d * h * w
                    projsrc = projsrc.permute(0, 3, 4, 2, 1).reshape(B, -1, 3, patcht_size, patch_size, patch_size)
                    projtar = extract_3Dpatches(tar_crop, patch_size, patcht_size, stride, stridet)  # b, c, d, h, w
                    projtar = projtar.permute(0, 3, 4, 2, 1).reshape(B, -1, 3, patcht_size, patch_size, patch_size)
                    nns = get_NN_indices_low_memory(projsrc, projtar, None, 1024)
                    projtar2src = projtar[torch.arange(B, device=nns.device)[:, None], nns]

                    err = (projtar2src - projsrc).abs().mean().item()
                    errs.append(err)

            return np.array(errs).mean()
