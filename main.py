import cv2
from cv2 import VideoCapture, imshow

cam_port = 0
cam = VideoCapture(cam_port, cv2.CAP_DSHOW)
result, image = cam.read()

if result:
    imshow("picture",image)
else:
    print("something wrong with pictures")

cam.release()
cv2.destroyAllWindows()
