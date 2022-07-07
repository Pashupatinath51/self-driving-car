
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utlis import *
import pandas as pd
  

#### STEP 1 - INITIALIZE DATA
path = r'C:\Users\shah9\Desktop\Project\Data Collection\DataCollected'

print('total length ', len(os.listdir(path)))
data = importDataInfo(path)
print(data.head())

data.head()

#### STEP 2 - VISUALIZE AND BALANCE DATA
data = balanceData(data,display=True)

#### STEP 3 - PREPARE FOR PROCESSING
imagesPath, steerings, sign = loadData(path,data)
#df = pd.DataFrame(zip(imagesPath, sign),
               #columns =['Image Path', 'Sign'])

#print('No of Path Created for Images ',len(imagesPath),len(steerings),len(sign))
#print('vvvvvv   ',steerings[5])
#print('sign  ', sign[5])

#### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
#steerings = float(steerings[-1])
#print(steerings)
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                              test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

#### STEP 5 - AUGMENT DATA

#### STEP 6 - PREPROCESS

#### STEP 7 - CREATE MODEL
model = createModel()

#### STEP 8 - TRAINNING
batch = int(len(xTrain)/100)
batch1 = int(len(xVal)/50)

history = model.fit(dataGen(xTrain, yTrain, batch, 1),
                                  steps_per_epoch=100,
                                  epochs=10,
                                  validation_data=dataGen(xVal, yVal, batch1, 0),
                                  validation_steps=50)

#### STEP 9 - SAVE THE MODEL
model.save('model.h5')
print('Model Saved')

#### STEP 10 - PLOT THE RESULTS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
