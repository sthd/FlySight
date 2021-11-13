import os.path
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import auxFunctions as aux
from ClipProcessing.all_clips_decor import all_clips
from EMD.EMD import make_basic_emd
from flyConvol import PhotoreceptorImageConverter

BUFFER_SIZE = 120

OBJECTS = {
    'A': 'Pillar',
    'B': 'Corner',
    'C': 'Wall Edge'
}

TEXTURES = ("Solid", "Checker Board", "Natural")

CF = {'Y': 0, 'N': 1}


def tranlate_clip_name(clip_file_name: str) -> str:
    pure_name = os.path.splitext(clip_file_name)[0]
    fields = pure_name.split('_')
    return f"{OBJECTS[fields[0]]}, Object Texture: {TEXTURES[int(fields[1]) - 1]},\n" \
           f" BG Texture: {TEXTURES[int(fields[2]) - 1]}, Movement {fields[3]} {'No' * CF[fields[5]]} Chicken Fence"


def basic_response_mid_horizontal(output_dir: str, input_clip: str):
    clip_file_name = get_clip_file_name(input_clip)
    greyscale_frames = capture_video_clip_frames(input_clip)
    angle_response_over_time_array = get_emd_responses(greyscale_frames, PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), greyscale_frames[0].shape, 6000))
    pickle_output_array(angle_response_over_time_array, clip_file_name, output_dir)
    save_surface_plot(angle_response_over_time_array, clip_file_name, output_dir)


def get_clip_file_name(input_clip: str) -> str:
    clip_file_name = os.path.split(input_clip)[1]
    print(clip_file_name)
    return clip_file_name


def get_emd_responses(g_frames, photoreceptor: PhotoreceptorImageConverter) -> np.array:
    angle_respone_over_time = []
    for buffer in photoreceptor.stream(g_frames, buffer_size=BUFFER_SIZE):
        emd = emd_row(buffer, buffer[0].shape[0] // 2)
        frequency_response_emd = [np.abs(np.fft.rfft(tr)) for tr in emd]
        angle_response_emd = angle_response_from_frequency_response_array(frequency_response_emd)
        angle_respone_over_time.append(angle_response_emd)
    art_arr = np.array(angle_respone_over_time)
    return art_arr


def capture_video_clip_frames(input_clip: str):
    cap = cv2.VideoCapture(input_clip)
    g_frames = []
    while True:
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
            cap.release()
            cv2.destroyAllWindows()
            break
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_frames.append(grayFrame)
    return g_frames


def pickle_output_array(output_array: np.array, clip_file_name: str, output_dir: str):
    with open(f"{os.path.join(output_dir, os.path.basename(clip_file_name))}.surface", 'wb') as sur:
        pickle.dump(output_array, sur)


def save_surface_plot(output_array: np.array, clip_file_name: str, output_dir: str):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    FRAMES, RESPONSES = output_array.shape
    x = np.array(range(FRAMES))
    y = np.array(range(RESPONSES))
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(np.transpose(X), np.transpose(Y), output_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('time [frames]')
    ax.set_ylabel('EMD')
    ax.set_zlabel('Amplitude')
    # Customize the z axis.
    ax.set_zlim(-.1, .5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_title(tranlate_clip_name(clip_file_name))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, os.path.splitext(clip_file_name)[0] + '.png'))
    print("Done")


def emd_row(buf, row_index):
    some_row = np.array(buf)[:, row_index, :]
    return [make_basic_emd()(some_row[:, i], some_row[:, i + 1]) for i in range(some_row.shape[1] - 1)]


def angle_response_from_frequency_response_array(fr_array):
    angle_response_emd = list()
    for fr in fr_array:
        integrand = list()
        for idx, val in enumerate(fr):
            if idx:
                normalizer = idx ** -2
            else:
                normalizer = 0
            integrand.append(normalizer * val)
        angle_response_emd.append(sum(integrand))
    return angle_response_emd


if __name__ == '__main__':
    all_clips(basic_response_mid_horizontal)
