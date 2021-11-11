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
from EMD.dual_signal_processor import DualSignalProcessor
from EMD.oned_filters import ButterworthLPF
from flyConvol import PhotoreceptorImageConverter

BUFFER_SIZE = 120


def emd_action(sig1, sig2, lpf, mul, sub):
    sig1_lpf = lpf(sig1)
    sig2_lpf = lpf(sig2)
    cross1 = mul(sig1_lpf, sig2)
    cross2 = mul(sig1, sig2_lpf)
    return sub(cross1, cross2)


LPF = ButterworthLPF()
MUL = DualSignalProcessor(np.multiply)
SUB = DualSignalProcessor(np.subtract)

EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)

test_EMD = DualSignalProcessor(emd_action, lpf=LPF, mul=MUL, sub=SUB)
EXAMPLE = "/Users/iddobar-haim/Library/Mobile Documents/com~apple~CloudDocs/FlySightProject/RealInputClips/Corner(B)/B_1_1/B_1_1_1_2_N_24_108.mp4"

OBJECTS = {
    'A': 'Pillar',
    'B': 'Corner',
    'C': 'Wall Edge'
}

TEXTURES = ("Solid", "Checker Board", "Natural")

CF = {
    'Y': 0,
    'N': 1
}


def tranlate_clip_name(clip_file_name: str) -> str:
    pure_name = os.path.splitext(clip_file_name)[0]
    fields = pure_name.split('_')
    return f"{OBJECTS[fields[0]]}, Object Texture: {TEXTURES[int(fields[1]) - 1]},\n" \
           f" BG Texture: {TEXTURES[int(fields[2]) - 1]}, Movement {fields[3]} {'No' * CF[fields[5]]} Chicken Fence"


def basic_response_mid_horizontal(output_dir, input_clip=EXAMPLE):
    clip_file_name = os.path.split(input_clip)[1]
    print(clip_file_name)
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
    pr = PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), g_frames[0].shape, 6000)
    art = []
    for buffer in pr.stream(g_frames, buffer_size=BUFFER_SIZE):
        emd = emd_row(buffer, buffer[0].shape[0] // 2)
        fr_emd = [np.abs(np.fft.rfft(tr)) for tr in emd]
        ar_emd = angle_response_from_frequency_response_array(fr_emd)
        art.append(ar_emd)
    art_arr = np.array(art)

    with open(f"{os.path.join(output_dir, os.path.basename(clip_file_name))}.surface", 'wb') as sur:
        pickle.dump(art_arr, sur)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    FRAMES, RESPONSES = art_arr.shape
    x = np.array(range(FRAMES))
    y = np.array(range(RESPONSES))
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(np.transpose(X), np.transpose(Y), art_arr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('time [frames]')
    ax.set_ylabel('EMD')
    ax.set_zlabel('Amplitude')
    # Customize the z axis.
    ax.set_zlim(-.1, 2.5)
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
    angle_response_emd = []
    for fr in fr_array:
        integrand = []
        for idx, val in enumerate(fr):
            if idx:
                normalizer = idx ** -2
            else:
                normalizer = 0
            integrand.append(normalizer * val)
        angle_response_emd.append(sum(integrand))
    return angle_response_emd

if __name__ == '__main__':
    # basic_response_mid_horizontal("/Users/iddobar-haim/Library/Mobile Documents/com~apple~CloudDocs/FlySightProject/RealInputClips/Corner(B)/B_1_1")
    all_clips(basic_response_mid_horizontal)
    # x=42

