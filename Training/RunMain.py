import cv2
import numpy as np
from tensorflow.keras.models import load_model

#import WebcamModule as wM
#import MotorModule as mM

#######################################
#steeringSen = 0.70  # Steering Sensitivity
#maxThrottle = 0.22  # Forward Speed %
#motor = mM.Motor(2, 3, 4, 17, 22, 27) # Pin Numbers
model = load_model(r'model.h5')
######################################

def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

cap = cv2.VideoCapture("track1.mp4")

while True:
    success, img = cap.read()
    img = cv2.resize(img, (240, 120))
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    steering = float(model.predict(img))
    print('string value is ',steering)

    cv2.waitKey(1)
    #motor.move(maxThrottle,-steering*steeringSen)
    # if cv2.waitKey(1) == 'q':
    #     break
    # cap.release()
    # cv2.destroyAllWindows()
