import glob
import os

import cv2
import numpy as np
import auxFunctions as aux
from flyConvol import PhotoreceptorImageConverter


def makeMissingDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def videoCodecAndExt(extension: str, scene_name):
    outputVideoPath = 'alternativeVideo'
    makeMissingDir(outputVideoPath)

    if extension == 'avi':
        output_video = './' + outputVideoPath + '/' + str(scene_name) + '.avi'
        output_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if extension == 'mp4':
        output_video = './' + outputVideoPath + '/' + str(scene_name) + '.mp4'
        output_codec = cv2.VideoWriter_fourcc(*'mp4v')
    return output_video, output_codec


def extractFPS(video_input):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video_input.get(cv2.cv.CV_CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video_input.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    return fps


# returns list of greyscale frames as numpy arrays
def extractVideoParameters(videoInput):
    return extractFPS(videoInput), int(videoInput.get(3)), int(videoInput.get(4))


def import_all_scenes():
    WHITE_LEVEL = 255
    scenes_directory = './videoScenes/'
    ext = ['avi', 'mp4', 'mov']  # video format extensions to be read
    scenes_files = []
    [scenes_files.extend(glob.glob(scenes_directory + '*.' + e)) for e in ext]
    return scenes_files


def break_into_frames():
    scenes_files=import_all_scenes()
    return scenes_files

#receives a numpy array and converts it to a normalised greyscale cv2 image
def convert_photoreceptors_to_cv2(image):
    image = 255 * image / np.max(image)
    cv2_image = image.astype(np.uint8)
    return cv2_image


def vidsss():
    ker = aux.make_gaussian_kernel(9, .5)
    scenes_files = import_all_scenes()
    for scene_file in scenes_files:
        all_frames = list()
        scene_video = cv2.VideoCapture(scene_file)
        scene_name = os.path.splitext(os.path.basename(scene_file))[0]
        fps, frame_width, frame_height = extractVideoParameters(scene_video)



        outputVideo, outputCodec = videoCodecAndExt('mp4', scene_name)
        outputVideo1, outputCodec = videoCodecAndExt('mp4', scene_name + str(1))
        out_after = cv2.VideoWriter(outputVideo1, outputCodec, fps, (103, 58))
        out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, frame_height))

        current_frame = 0
        while True:
            ret, frame = scene_video.read()
            if not ret:
                break
            framesPath = 'frames/' + str(scene_name)
            makeMissingDir(framesPath)
            name = './' + framesPath + '/frame' + str(current_frame) + '.jpg'
            grey_frame_2D = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey_frame_3D = cv2.cvtColor(grey_frame_2D, cv2.COLOR_GRAY2BGR)

            pr = PhotoreceptorImageConverter(ker, grey_frame_2D.shape, 750 * 8)
            frame_post_convol_2D=convert_photoreceptors_to_cv2(pr.apply(grey_frame_2D))
            frame_post_convol_3D = cv2.cvtColor(frame_post_convol_2D, cv2.COLOR_GRAY2BGR)
            out.write(grey_frame_3D)  # video is constructed of 3d arrays
            out_after.write(frame_post_convol_3D)
            all_frames.append(frame_post_convol_2D)
            # print('Creating...' + name)
            # aux.greyscale_plot(a2_greyFrame3D)
            cv2.imwrite(name, frame_post_convol_3D)  # save grey scale frames of each video
            # cv2.imshow('video gray', greyFrame2D)
            # cv2.waitKey(0)
            current_frame += 1

        scene_video.release()
        out.release()
        out_after.release()
        cv2.destroyAllWindows()
        return all_frames

if __name__ == '__main__':
    numpy_frames =vidsss()
    print(numpy_frames)







        #counter = 0
        # if counter < 4:
        # counter += 1

        # print(frame_res_int)

        # aux.greyscale_plot(res)
        # aux.greyscale_plot(greyFrame2D)

        # a1_greyFrame2D = cv2.cvtColor(frame_res_int, cv2.COLOR_BGR2GRAY)

        # return allFrames
    # print(allFrames[1])

# res /=255

# dev1 = (res / WHITE_LEVEL)
# print(greyFrame2D.shape)

# print(type(jd))
# print(jd.shape)
# print(jd.ndim)
# print(jd)


    # breakpoint()


# fps=10
# frame_width=103
# frame_height=58