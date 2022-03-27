import glob
import os

import cv2
import numpy as np
import auxFunctions as aux
from flyConvol import PhotoreceptorImageConverter

import sys
np.set_printoptions(threshold=sys.maxsize)

# image = cv2.imread('pic1.png')
# original_array = np.loadtxt("test.txt").reshape(4, 2)
# print(original_array)

def draw_synthetic_circle(fps, video_length, x_centre, y_centre, radius, bgr):
    shape_drawn = 'rect'
    #shape_drawn = 'circle'
    allFrames = list()
    current_frame = 0
    output_video = './data/synthVideos/synthVid_' + str(shape_drawn) +'.mp4'
    output_codec = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width=1920
    frame_height=1080
    synthOut = cv2.VideoWriter(output_video, output_codec, fps, (frame_width, frame_height))
    x_centre = 0
    while current_frame < fps*video_length:
        current_frame += 1
        #img = np.zeros([100, 200, 3], dtype=np.uint8)
        img = np.zeros([1080, 1920, 3], dtype=np.uint8)
        img.fill(255)  # or img[:] = 255

        #name='synthetic_frame' + str(current_frame) + '.jpg'


        #cv2.circle(img,(x_centre, y_centre), radius, bgr , -1)

        cv2.rectangle(img,(x_centre-250, y_centre-300),(x_centre, y_centre+500), color=bgr, thickness=-1)

        synth_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        name = './data/synthFrames/synth_' + str(shape_drawn) + '/'+ str(shape_drawn) + str(current_frame) + '.jpg'
        cv2.imwrite(name, synth_img)

        synth_img3D = cv2.cvtColor(synth_img, cv2.COLOR_GRAY2BGR)
        #print(test_image)
        allFrames.append(synth_img)

        #image = cv2.imread(name)
        #print(test_image)
        #cv2.imshow(str(current_frame), test_image)
        #cv2.waitKey(0)
        #x_centre+=3
        x_centre += 15
        cv2.destroyAllWindows()
        synthOut.write(synth_img3D)

    synthOut.release()
    cv2.destroyAllWindows()
    return allFrames


from ClipProcessing.basic_process import basic_response_mid_horizontal

def makeMissingDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def videoCodecAndExt(extension: str, scene_name):
    outputVideoPath = 'data/alternativeVideo'
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
    scenes_directory = './data/videoScenes/'
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


def vids_save_mid_frame():
    ker = aux.make_gaussian_kernel(9, .5)
    scenes_files = import_all_scenes()

    for scene_file in scenes_files:
        all_frames = list()
        scene_video = cv2.VideoCapture(scene_file)
        scene_name = os.path.splitext(os.path.basename(scene_file))[0]
        fps, frame_width, frame_height = extractVideoParameters(scene_video)
        outputVideo, outputCodec = videoCodecAndExt('mp4', scene_name + "_highlight")
        #basic_response_mid_horizontal("/Users/elior/PycharmProjects/FlySight/data/graphs", scene_name )
        out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, frame_height))
        #print(frame_width)  #1920
        #print(frame_height) #1080
        current_frame = 0
        while True:
            ret, frame = scene_video.read()
            if not ret:
                break
            framesPath = 'data/frames/' + str(scene_name)
            makeMissingDir(framesPath)
            name = './' + framesPath + '/frame' + str(current_frame) + '.jpg'
            grey_frame_2D = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey_frame_3D = cv2.cvtColor(grey_frame_2D, cv2.COLOR_GRAY2BGR)

            pr = PhotoreceptorImageConverter(ker, grey_frame_2D.shape, 750 * 8)
            frame_post_convol_2D = convert_photoreceptors_to_cv2(pr.apply(grey_frame_2D))
            frame_post_convol_3D = cv2.cvtColor(frame_post_convol_2D, cv2.COLOR_GRAY2BGR)

            mid_frame = frame_height // 2
            crop_mid = grey_frame_3D[mid_frame: mid_frame + 11, 0:frame_width]
            cv2.imwrite(name, crop_mid)  # save grey scale mid frames of each video

            grey_frame_3D[mid_frame - 1, :] = [[0, 0, 200]] * grey_frame_3D.shape[1]
            grey_frame_3D[mid_frame + 10, :] = [[0, 0, 200]] * grey_frame_3D.shape[1]
            out.write(grey_frame_3D)  # video is constructed of 3d arrays

            all_frames.append(frame_post_convol_2D)
            # print('Creating...' + name)
            # aux.greyscale_plot(a2_greyFrame3D)

            # cv2.imshow('video gray', greyFrame2D)
            # cv2.waitKey(0)
            current_frame += 1
        scene_video.release()
        out.release()
        cv2.destroyAllWindows()
    return all_frames


#def highlight_save_mid(input_clip, ker):



if __name__ == '__main__':
    #numpy_frames =vids_save_mid_frame()
    print("Done")

    #image = cv2.imread('pic1.png')
    #image='pic1.png'
    bgr=(0,0,0)
    draw_synthetic_circle(30, 4.6, -90, 540, 90, bgr) #(fps, video_length, x_centre, y_centre, radius, bgr):





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