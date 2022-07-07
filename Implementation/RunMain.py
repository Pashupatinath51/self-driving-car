import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


#import WebcamModule as wM
#import MotorModule as mM

#######################################
#steeringSen = 0.70  # Steering Sensitivity
#maxThrottle = 0.22  # Forward Speed %
#motor = mM.Motor(2, 3, 4, 17, 22, 27) # Pin Numbers
model = load_model(r'C:\Users\shah9\Desktop\Project\Implementation\model.h5')
######################################
def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

model1 = tf.keras.models.load_model(r'C:\Users\shah9\Desktop\Project\Data Collection\my_model')


if __name__ == '__main__':
    
    cap = cv2.VideoCapture(r"C:\Users\shah9\Desktop\Project\Data Collection\lane.mp4")
    
    frameCounter = 0
    while (cap.isOpened()):
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
    
        ret, img = cap.read()
        frame = img.copy()
        frame1 = img.copy()
        
        img = cv2.resize(img, (240, 120))
        img = np.asarray(img)
        img = preProcess(img)
        #img = img.reshape(180,180,3)
        img = np.array([img])
        steering = float(model.predict(img))
        print("Steering Value :- ",steering)
        img = np.asarray(frame)
        img = cv2.resize(img, (180, 180))
    
        img = img.reshape(180,180,3)
        img = np.array([img])
        predictions = model1.predict(img)
        probVal =int(np.argmax(predictions,axis=1))
        print(probVal)
        cv2.imshow('frame',frame1)
    #print(steering*steeringSen)
    #motor.move(maxThrottle,-steering*steeringSen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()    