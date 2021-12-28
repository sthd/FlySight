import glob
import os

import cv2
import numpy as np
import auxFunctions as aux
from flyConvol import PhotoreceptorImageConverter


def makeMissingDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def videoCodecAndExt(extension: str):
    if extension == 'avi':
        output_video = './' + outputVideoPath + '/' + str(sceneName) + '.avi'
        output_codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if extension == 'mp4':
        output_video = './' + outputVideoPath + '/' + str(sceneName) + '.mp4'
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

#returns list of greyscale frames as numpy arrays
def extractVideoParameters(videoInput):
    return extractFPS(videoInput), int(videoInput.get(3)), int(videoInput.get(4))

#def import_all_scenes():

if __name__ == '__main__':

    ker = aux.make_gaussian_kernel(9, .5)
    #breakpoint()

    WHITE_LEVEL = 255
    videoDir = './videoScenes/'
    ext = ['avi', 'mp4', 'mov']  # video format extensions to be read
    videoFiles = []
    [videoFiles.extend(glob.glob(videoDir + '*.' + e)) for e in ext]

    for file in videoFiles:
        allFrames = list()
        videoInput = cv2.VideoCapture(file)
        sceneName = os.path.splitext(os.path.basename(file))[0]
        fps, frame_width, frame_height = extractVideoParameters(videoInput)
        #fps=10
        #frame_width=103
        #frame_height=58
        outputVideoPath = 'alternativeVideo'
        makeMissingDir(outputVideoPath)

        outputVideo, outputCodec = videoCodecAndExt('mp4')
        ###out_after = cv2.VideoWriter(outputVideo, outputCodec, fps, (103, 58))
        out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, 10))
        counter = 0
        currentFrame = 0
        while True:
            ret, frame = videoInput.read()
            if not ret:
                break
            framesPath = 'frames/' + str(sceneName)
            makeMissingDir(framesPath)
            name = './' + framesPath + '/frame' + str(currentFrame) + '.jpg'

            mid_frame = frame_height // 2
            crop_img = frame[mid_frame : mid_frame + 10 , 0:frame_width]
            print(crop_img.shape[0])
            print("yay")
            print(crop_img.shape[1])
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey(0)

            a_greyFrame2D = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            ###a_greyFrame2D = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            a_greyFrame3D = cv2.cvtColor(a_greyFrame2D, cv2.COLOR_GRAY2BGR)


            pr = PhotoreceptorImageConverter(ker, a_greyFrame2D.shape, 750*8)
            frame_res = pr.apply(a_greyFrame2D)

            frame_res = 255 * frame_res / np.max(frame_res)
            frame_res_int = frame_res.astype(np.uint8)

            a2_greyFrame3D = cv2.cvtColor(frame_res_int, cv2.COLOR_GRAY2BGR)
            out.write(a2_greyFrame3D)  # video is constructed of 3d arrays
            #allFrames.append(a_greyFrame2D)
            # print('Creating...' + name)
            #aux.greyscale_plot(a2_greyFrame3D)
            ###cv2.imwrite(name, a2_greyFrame3D)  # save grey scale frames of each video

            cv2.imwrite(name, crop_img)  # save grey scale FRAMES of each video

            # cv2.imshow('video gray', greyFrame2D)
            # cv2.waitKey(0)
            currentFrame += 1

        videoInput.release()
        out.release()
        cv2.destroyAllWindows()

        # if counter < 4:
        # counter += 1


        # print(frame_res_int)

        # aux.greyscale_plot(res)
        # aux.greyscale_plot(greyFrame2D)

        # a1_greyFrame2D = cv2.cvtColor(frame_res_int, cv2.COLOR_BGR2GRAY)

        #return allFrames
    # print(allFrames[1])

# res /=255

# dev1 = (res / WHITE_LEVEL)
# print(greyFrame2D.shape)

# print(type(jd))
# print(jd.shape)
# print(jd.ndim)
# print(jd)