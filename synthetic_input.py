import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


# image = cv2.imread('pic1.png')
# original_array = np.loadtxt("test.txt").reshape(4, 2)
# print(original_array)

def draw_synthetic_circle(fps, video_length, x_centre, y_centre, radius, bgr):
    allFrames = list()
    current_frame = 0
    while current_frame < fps*video_length:
        current_frame += 1
        img = np.zeros([100, 200, 3], dtype=np.uint8)
        img.fill(255)  # or img[:] = 255

        #name='synthetic_frame' + str(current_frame) + '.jpg'
        cv2.circle(img,(x_centre, y_centre), radius, bgr , -1)
        test_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(test_image)
        allFrames.append(test_image)
        name = './framesis' + str(current_frame) + '.jpg'
        cv2.imwrite(name, test_image)
        image = cv2.imread(name)
        print(test_image)
        cv2.imshow(str(current_frame), test_image)
        cv2.waitKey(0)
        x_centre+=3
    return allFrames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #image = cv2.imread('pic1.png')
    image='pic1.png'
    bgr=(0,0,0)
    draw_synthetic_circle(30, 5, 26, 27, 25, bgr)

