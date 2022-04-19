import cv2 as cv
import numpy as np
from Classifier import DogImageClassifier

root_wind = 'main'
cv.namedWindow(root_wind)
dog = DogImageClassifier()
# Create a black image
img = np.ones((600, 1200, 3), np.uint8)
# img = cv.rectangle(img,(0,0),(1200,30),(50,50,50),-1)
# img = cv.rectangle(img,(0,0),(30,600),(50,50,50),-1)
# user cam box
img = cv.rectangle(img, (30, 30), (529, 429), (255, 50, 50), -1)
# Start button
img = cv.rectangle(img, (550, 70), (630, 100), (50, 50, 50), -1)
# Screenshot button
img = cv.rectangle(img, (550, 140), (630, 170), (50, 50, 50), -1)
# Dog picture
img = cv.rectangle(img, (650, 30), (1149, 429), (255, 50, 50), -1)
# Textbox
img = cv.rectangle(img, (30, 450), (1150, 550), (50, 50, 50), -1)

font = cv.FONT_HERSHEY_SIMPLEX
# start text
img = cv.putText(img, 'Start', (551, 94), font, 1, (0, 0, 255), 1)
# capture text
img = cv.putText(img, 'Capture', (554, 163), font, 0.6, (0, 0, 255), 1)
result = None
image = None
close_dog = None
cam_port = 0
cam = cv.VideoCapture(cam_port, cv.CAP_DSHOW)
text_string = None

cam_condition = False


class Placeholder:

    def __init__(self):
        self.condition = None
        self.fetch_dog = False


place = Placeholder()


def print_mouse(event, x, y, flags, param):
    place = param["placeholder"]
    if event == cv.EVENT_LBUTTONDOWN:
        if 550 < x < 630 and 70 < y < 100:
            place.condition = True
        if 550 < x < 630 and 140 < y < 170:
            place.condition = False
            place.fetch_dog = True


cv.setMouseCallback("main", print_mouse, param={"placeholder": place})

while True:

    # keyboard events
    code = cv.waitKey(1)

    if code == ord('q'):
        break

    if place.condition == True:
        result, image = cam.read()

    #        cv.imshow(image)
    #       print(image.shape)

    if result and place.condition:
        image = cv.resize(image, (500, 400))
        img[30:430, 30:530] = image

    if place.fetch_dog == True:
        place.fetch_dog = False
        if result:
            close_dog = dog.get_close_image(image)

    if close_dog is not None:
        img = cv.rectangle(img, (30, 450), (1150, 550), (50, 50, 50), -1)
        img[30:430, 650:1150] = close_dog
        text_string = f"This image looks like a {dog.classification} with {round(dog.classification_score * 100,2)}% likelihood"
        img = cv.putText(img, text_string, (100, 500), font, 1, (0, 255, 0), 1)

    cv.imshow(root_wind, img)

cv.destroyAllWindows()

# cam_port = 0
# cam = VideoCapture(cam_port, cv2.CAP_DSHOW)
# result, image = cam.read()


# cam.release()
