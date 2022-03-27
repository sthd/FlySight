import glob
import os
import pickle

from matplotlib import pyplot as plt

from ClipProcessing.all_clips_decor import OBJECT_DIRS, ROOT
from ClipProcessing.helping_functions import tranlate_clip_name
from tqdm import tqdm

if __name__ == '__main__':
    for odir in OBJECT_DIRS:
        for surface_file_path in tqdm(glob.glob(os.path.join(ROOT, odir, '*', '*', '*.surface'))):
            surface_file_name = os.path.splitext(os.path.split(surface_file_path)[1])[0]
            surface_function_name = os.path.split(os.path.split(surface_file_path)[0])[1]
            with open(surface_file_path, "rb") as sur:
                output_mat = pickle.load(sur)

            fig, ax = plt.subplots()
            im = ax.imshow(output_mat)
            ax.set_title(tranlate_clip_name(surface_file_name) + "\n" + surface_function_name)
            plt.savefig(os.path.splitext(surface_file_path)[0] + "_colormap.png")
            plt.close('all')
