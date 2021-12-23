import os
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator

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


def get_clip_file_name(input_clip: str) -> str:
    clip_file_name = os.path.split(input_clip)[1]
    print(clip_file_name)
    return clip_file_name


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
    ax.set_zlim(np.min(output_array) - 0.1*np.abs(np.min(output_array)), np.max(output_array) + 0.1*np.abs(np.max(output_array)))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_title(tranlate_clip_name(clip_file_name))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, os.path.splitext(clip_file_name)[0] + '.png'))
    plt.close('all')
    print("Done")
