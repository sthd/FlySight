import cv2
import numpy as np



def draw_synthetic_circle(fps, video_length, x_centre, y_centre, radius, bgr, image):
    allFrames = list()
    current_frame = 0
    while current_frame < fps*video_length:
        current_frame += 1
        #image = cv2.imread('pic1.png')
        img = np.zeros([1000, 2000, 3], dtype=np.uint8)
        img.fill(255)  # or img[:] = 255

        #name='synthetic_frame' + str(current_frame) + '.jpg'
        cv2.circle(img,(x_centre, y_centre), radius, bgr , -1)
        test_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        allFrames.append(test_image)
        cv2.imshow(str(current_frame),test_image)
        cv2.waitKey(0)
        x_centre+=3
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread('pic1.png')
    #image='pic1.png'
    bgr=(0,0,0)
    draw_synthetic_circle(30, 5, 100, 30, 25, bgr, image)

