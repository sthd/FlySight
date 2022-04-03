import glob
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from ClipProcessing.all_clips_decor import OBJECT_DIRS, ROOT_IDDO, ROOT_ELIOR
from ClipProcessing.helping_functions import tranlate_clip_name
from tqdm import tqdm

N = 20

if __name__ == '__main__':
    for odir in OBJECT_DIRS:
        for surface_file_path in tqdm(glob.glob(os.path.join(ROOT_IDDO, odir, '*', '*', '*.surface'))):
            surface_file_name = os.path.splitext(os.path.split(surface_file_path)[1])[0]
            surface_function_name = os.path.split(os.path.split(surface_file_path)[0])[1]
            with open(surface_file_path, "rb") as sur:
                output_mat = pickle.load(sur)

            fig, axs = plt.subplots(nrows=2, ncols=1)
            im = axs[0].imshow(output_mat)
            axs[0].set_title(tranlate_clip_name(surface_file_name) + "\n" + surface_function_name)
            axs[0].set_xlabel("Angular Response")
            axs[0].set_ylabel("Frame Number")

            frame_num = output_mat.shape[0] // 2
            im2 = axs[1].plot(np.convolve(output_mat[frame_num, :], np.ones(N) / N, mode='valid'))
            axs[1].set_title(f"Angular response at {frame_num=}")
            plt.savefig(os.path.splitext(surface_file_path)[0] + "_colormap.png")
            plt.close('all')
