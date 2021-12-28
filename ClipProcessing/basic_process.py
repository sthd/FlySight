import numpy as np
import pickle
import auxFunctions as aux
from ClipProcessing.all_clips_decor import all_clips
from ClipProcessing.helping_functions import capture_video_clip_frames, get_clip_file_name, pickle_output_array, save_surface_plot
# try_my_graph, try_my_graph2
#from ClipProcessing.helping_functions import try_my_graph as tmg
from EMD.EMD import make_basic_emd
from flyConvol import PhotoreceptorImageConverter

BUFFER_SIZE = 120


def basic_response_mid_horizontal(output_dir: str, input_clip: str):
    """
    Calculates the angle response of an array of EMDs located at the the medial horizontal line of each frame.
    Inspired by the paper - "Spatial Encoding of Translational Optic Flow in Planar Scenes by Elementary Motion Detector Arrays".
    :param output_dir: The directory to save the output(s) to.
    :param input_clip: The clip to process.
    """
    clip_file_name = get_clip_file_name(input_clip)
    greyscale_frames = capture_video_clip_frames(input_clip)
    angle_response_over_time_array = get_emd_responses(greyscale_frames,
                                                       PhotoreceptorImageConverter(aux.make_gaussian_kernel(15), greyscale_frames[0].shape, 6000))
    pickle_output_array(angle_response_over_time_array, clip_file_name, output_dir)
    save_surface_plot(angle_response_over_time_array, clip_file_name, output_dir)


def get_emd_responses(frames, photoreceptor: PhotoreceptorImageConverter) -> np.array:
    """
    Calculate The angle responses through the photoreceptor's.
    :param frames: The frames to process.
    :param photoreceptor: The converter of frames to photoreceptor responses.
    :return: The angle response array for each frame.
    """
    angle_respone_over_time = []
    for buffer in photoreceptor.stream(frames, buffer_size=BUFFER_SIZE):
        emd = emd_row(buffer, buffer[0].shape[0] // 2)
        frequency_response_emd = [np.abs(np.fft.rfft(tr)) for tr in emd]
        angle_response_emd = angle_response_from_frequency_response_array(frequency_response_emd)
        angle_respone_over_time.append(angle_response_emd)
    return np.array(angle_respone_over_time)


def emd_row(buf, row_index):
    chosen_horizontal_line = np.array(buf)[:, row_index, :]
    return [make_basic_emd()(chosen_horizontal_line[:, i], chosen_horizontal_line[:, i + 1]) for i in range(chosen_horizontal_line.shape[1] - 1)]


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
    #all_clips(basic_response_mid_horizontal)
    #A_1_3_1_2_N_24_80_Cut
    #A_1_3_1_2_N_24_80_Cut.mp4.surface
    path="/Users/elior/Library/Mobile Documents/com~apple~CloudDocs/FlySightProject/RealInputClips2/Pillar(A)/A_1_3/basic_response_mid_horizontal_func/A_1_3_1_2_N_24_80_Cut.mp4.surface"
    pickle_in = open(f"{path}", 'rb')
    example = pickle.load(pickle_in)
    #save_surface_plot(angle_response_over_time_array, clip_file_name, output_dir)

    #try_my_graph(example, "1.mp4", "/Users/elior/Library/Mobile Documents/")
    #try_my_graph2(example, "1.mp4", "/Users/elior/Library/Mobile Documents/")
    #print(example)
