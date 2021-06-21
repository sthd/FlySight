import glob
import os

import cv2

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

        outputVideoPath = 'alternativeVideo'
        makeMissingDir(outputVideoPath)

        outputVideo, outputCodec = videoCodecAndExt('mp4')
        # out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, frame_height))
        out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, frame_height))
        counter = 0
        currentFrame = 0
        while True:
            ret, frame = videoInput.read()
            if not ret:
                break
            framesPath = 'frames/' + str(sceneName)
            makeMissingDir(framesPath)
            name = './' + framesPath + '/frame' + str(currentFrame) + '.jpg'
            greyFrame2D = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            greyFrame3D = cv2.cvtColor(greyFrame2D, cv2.COLOR_GRAY2BGR)

            #if counter < 4:
                #counter += 1
            pr = PhotoreceptorImageConverter(ker, greyFrame2D.shape, 750*8)
            res = pr.apply(greyFrame2D)
            print(res)
            res = res / 255
            res_int = res.astype(int)
                #dev1 = (res / WHITE_LEVEL)
                #print(greyFrame2D.shape)

                # print(type(jd))
                # print(jd.shape)
                # print(jd.ndim)
                # print(jd)
            aux.greyscale_plot(res)
            print(res_int)
            aux.greyscale_plot(greyFrame2D)

            #greyFrame2D = cv2.cvtColor(res_int, cv2.COLOR_BGR2GRAY)
            greyFrame3D = cv2.cvtColor(res_int, cv2.COLOR_GRAY2BGR)
            out.write(greyFrame3D)  # video is constructed of 3d arrays
            allFrames.append(greyFrame2D)
            # print('Creating...' + name)
            cv2.imwrite(name, greyFrame2D)  # save grey scale frames of each video
            # cv2.imshow('video gray', greyFrame2D)
            # cv2.waitKey(0)
            currentFrame += 1

        videoInput.release()
        out.release()
        cv2.destroyAllWindows()
        #return allFrames
    # print(allFrames[1])
