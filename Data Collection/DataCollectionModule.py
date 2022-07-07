"""
- This module saves images and a log file.
- Images are saved in a folder.
- Folder should be created manually with the name "DataCollected"
- The name of the image and the steering angle is logged
in the log file.
- Call the saveData function to start.
- Call the saveLog function to end.
- If runs independent, will save ten images as a demo.
"""
import utlis
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from datetime import datetime
from lane import getLaneCurve
global imgList, steeringList, signdetection
countFolder = 0
count = 0
imgList = []
steeringList = []
signdetection = []
#GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'DataCollected')
# print(myDirectory)

# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
        countFolder += 1
newPath = myDirectory +"/IMG"+str(countFolder)
os.makedirs(newPath)

# SAVE IMAGES IN THE FOLDER
def saveData(img,steering,sign):
    global imgList, steeringList, signdetection
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    steeringList.append(steering)
    signdetection.append(sign)

# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog():
    global imgList, steeringList, signdetection 
    rawData = {'Image': imgList,
                'Steering': steeringList,
                'Sign' : signdetection }
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory,f'log_{str(countFolder)}.csv'), index=False, header=False)
    print('Log Saved')
    print('Total Images: ',len(imgList))

if __name__ == '__main__':
    cap = cv2.VideoCapture("track1.mp4")
    intialTrackBarVals = [41, 40, 3, 182]
    utlis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        curveVal = getLaneCurve(img, 2)

        saveData(img, curveVal,2)
        saveLog()
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()