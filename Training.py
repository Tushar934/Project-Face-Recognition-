import cv2
from os import listdir
from os.path import isfile,join
import numpy as np
print(cv2.__version__)

    #get the training data we perviously made
data_path='./images/'

onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

#creating arrays for training data and labels
Training_Data,Labels=[],[]

#open training images in our datapath
#create a numpy array for training data
for i,files in enumerate(onlyfiles):
    image_path=data_path+"/"+onlyfiles[i]
    images1=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images1,dtype=np.uint8))
    Labels.append(i)

#Create a numpy array for both training data and labels
Labels=np.asarray(Labels,dtype=np.int32)
tushar_model=cv2.face_LBPHFaceRecognizer.create()
#Intialize facial recognizer

#Let's train our model
tushar_model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Model Trained Succesfully")