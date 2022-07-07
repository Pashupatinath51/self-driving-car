import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import DataCollectionModule as dcM
from time import sleep
import utlis
import cv2
from lane import getLaneCurve

import numpy as np
import tensorflow as tf
from tensorflow import keras

#maxThrottle = 0.25
#motor = mM.Motor(2, 3, 4, 17, 22, 27)

class_names = ['Ahead', 'Footpath', 'Go', 'Stop', 'Turn', 'UTurn']
model = tf.keras.models.load_model(r'C:\Users\shah9\Desktop\Project\Data Collection\my_model')
if __name__ == '__main__':
    cap = cv2.VideoCapture("lane.mp4")
    intialTrackBarVals = [41, 40, 3, 182]
    utlis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img = cap.read()
        curveVal = getLaneCurve(img, 0)
        frame = img.copy()
        frame1 = img.copy()
        img = np.asarray(frame)
        img = cv2.resize(img, (180, 180))
    
        img = img.reshape(180,180,3)
        img = np.array([img])
        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])
        probVal =int(np.argmax(predictions,axis=1))
        print(probVal)
        if probVal > 0.2 :
            cv2.putText(frame,str(class_names[probVal]), (250,250), cv2.FONT_HERSHEY_COMPLEX,3, (0,0,255),3)
            cv2.namedWindow("processed", cv2.WINDOW_NORMAL)
        cv2.imshow("processed", frame)
            #cv2.imshow("Image", frame)
        dcM.saveData(frame1, curveVal,probVal)
        dcM.saveLog()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
