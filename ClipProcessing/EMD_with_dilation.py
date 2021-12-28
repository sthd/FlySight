import os
import pickle

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator

import auxFunctions as aux
from ClipProcessing.all_clips_decor import all_clips
from ClipProcessing.helping_functions import capture_video_clip_frames, get_clip_file_name, pickle_output_array, save_surface_plot, tranlate_clip_name
from EMD.EMD import make_basic_emd
from flyConvol import PhotoreceptorImageConverter

BUFFER_SIZE = 120


def pickle_output_array_d(output_array: np.array, clip_file_name: str, output_dir: str, d):
    with open(f"{os.path.join(output_dir, os.path.basename(clip_file_name))}_{str(d)}.surface", 'wb') as sur:
        pickle.dump(output_array, sur)


def save_surface_plot_d(output_array: np.array, clip_file_name: str, output_dir: str, d):
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
    ax.set_zlim(np.min(output_array) - 0.1 * np.abs(np.min(output_array)), np.max(output_array) + 0.1 * np.abs(np.max(output_array)))
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set_title(tranlate_clip_name(clip_file_name))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(clip_file_name)[0]}_{str(d)}.png"))
    plt.close('all')
    print("Done")


def     dilation_response_mid_horizontal(output_dir: str, input_clip: str, dilation):
    """
    Calculates the angle response of an array of EMDs located at the the medial horizontal line of each frame.
    Inspired by the paper - "Spatial Encoding of Translational Optic Flow in Planar Scenes by Elementary Motion Detector Arrays".
    :param dilation: dilation size.
    :param output_dir: The directory to save the output(s) to.
    :param input_clip: The clip to process.
    """
    clip_file_name = get_clip_file_name(input_clip)
    greyscale_frames = capture_video_clip_frames(input_clip)
    angle_response_over_time_array = get_emd_responses(greyscale_frames,
                                                       PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), greyscale_frames[0].shape, 6000), dilation)
    pickle_output_array_d(angle_response_over_time_array, clip_file_name, output_dir, dilation)
    save_surface_plot_d(angle_response_over_time_array, clip_file_name, output_dir, dilation)


def get_emd_responses(frames, photoreceptor: PhotoreceptorImageConverter, dilation) -> np.array:
    """
    Calculate The angle responses through the photoreceptor's.
    :param dilation: dilation size.
    :param frames: The frames to process.
    :param photoreceptor: The converter of frames to photoreceptor responses.
    :return: The angle response array for each frame.
    """
    angle_respone_over_time = []
    for buffer in photoreceptor.stream(frames, buffer_size=BUFFER_SIZE):
        emd = emd_row(buffer, buffer[0].shape[0] // 2, dilation)
        frequency_response_emd = [np.abs(np.fft.rfft(tr)) for tr in emd]
        angle_response_emd = angle_response_from_frequency_response_array(frequency_response_emd)
        angle_respone_over_time.append(angle_response_emd)
    return np.array(angle_respone_over_time)


def emd_row(buf, row_index, dilation):
    chosen_horizontal_line = np.array(buf)[:, row_index, :]
    return [make_basic_emd()(chosen_horizontal_line[:, i], chosen_horizontal_line[:, i + dilation]) for i in range(chosen_horizontal_line.shape[1] - dilation)]


def angle_response_from_frequency_response_array(frequency_response_array: np.array) -> np.array:
    """
    Calculates Equation (1) from the paper - "Spatial Encoding of Translational Optic Flow in Planar Scenes by Elementary Motion Detector Arrays".
    Namely, R_ø =∫R(ƒ)/ƒ^2dƒ
    :param frequency_response_array: R(ƒ)
    :return: R_ø
    """
    angle_response_emd = list()
    for fr in frequency_response_array:
        integrand = list()
        for idx, val in enumerate(fr):
            if idx:
                normalizer = idx ** -2  # 1/ƒ^2
            else:
                normalizer = 0  # Cancel DC response and prevent division by 0
            integrand.append(normalizer * val)
        angle_response_emd.append(sum(integrand))
    return angle_response_emd


if __name__ == '__main__':
    for d in range(1, 5):
        all_clips(dilation_response_mid_horizontal, dilation=d)
