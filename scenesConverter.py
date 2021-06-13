import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from PIL import Image as im
import scipy.signal
from PIL import Image , ImageFilter
import auxFunctions as aux
from flyConvol import PhotoreceptorImageConverter

def makeMissingDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def videoCodecAndExt(extension:str):
    if (extension == 'avi'):
        outputVideo = './' + outputVideoPath +'/' + str(sceneName) + '.avi'
        outputCodec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if (extension == 'mp4'):
        outputVideo = './' + outputVideoPath + '/' + str(sceneName) + '.mp4'
        outputCodec = cv2.VideoWriter_fourcc(*'mp4v')
    return outputVideo, outputCodec

def extractFPS(videoInput):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = videoInput.get(cv2.cv.CV_CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = videoInput.get(cv2.CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    return fps

def extractVideoParameters(videoInput):
    fps=extractFPS(videoInput)
    frame_width = int(videoInput.get(3))
    frame_height = int(videoInput.get(4))
    return fps, frame_width, frame_height

if __name__ == '__main__':

    ker = aux.make_gaussian_kernel(9, .5)

    WHITE_LEVEL = 255
    videoDir = './videoScenes/'
    ext = ['avi', 'mp4', 'mov'] # video format extensions to be read
    videoFiles = []
    [videoFiles.extend(glob.glob(videoDir + '*.' + e)) for e in ext]

    for file in videoFiles:
        allFrames=[]
        videoInput = cv2.VideoCapture(file)
        sceneName = os.path.splitext(os.path.basename(file))[0]
        fps, frame_width, frame_height = extractVideoParameters(videoInput)

        outputVideoPath='alternativeVideo'
        makeMissingDir(outputVideoPath)

        outputVideo, outputCodec = videoCodecAndExt('mp4')
        #out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, frame_height))
        out = cv2.VideoWriter(outputVideo, outputCodec, fps, (frame_width, frame_height))
        jj = 0
        currentFrame = 0
        while(True):
            ret, frame = videoInput.read()
            if not ret: break
            framesPath='frames/' + str(sceneName)
            makeMissingDir(framesPath)
            name = './' + framesPath +'/frame' + str(currentFrame) + '.jpg'

            greyFrame2D = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            greyFrame3D = cv2.cvtColor(greyFrame2D, cv2.COLOR_GRAY2BGR)

            #if (jj < 1 ):
                #jj+=1
            pr = PhotoreceptorImageConverter(ker, (greyFrame2D.shape[0], greyFrame2D.shape[1]), 2000)
            res = pr.apply(greyFrame2D)
            dev1 = (res / WHITE_LEVEL)

            jd = dev1.astype(int)
            print(type(jd))
            print(jd.shape)
            print(jd.ndim)
            print(jd)
            #aux.greyscale_plot(res)
            #aux.greyscale_plot(jd)

            greyFrame3D = cv2.cvtColor(greyFrame2D, cv2.COLOR_GRAY2BGR)
            out.write(greyFrame3D) #video is constructed of 3d arrays
            allFrames.append(greyFrame2D)
            #print('Creating...' + name)
            cv2.imwrite(name, greyFrame2D)      #save grey scale frames of each video
            #cv2.imshow('video gray', greyFrame2D)
            #cv2.waitKey(0)
            currentFrame+=1

        videoInput.release()
        out.release()
        cv2.destroyAllWindows()

    #print(allFrames[1])

